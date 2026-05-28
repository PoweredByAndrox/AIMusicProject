"""
ML Worker Main Entry Point

Starts the queue worker and registers task handlers.
Requirements: 13.1, 13.2, 13.3, 13.4
"""

import os
import sys
import logging
import signal
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import numpy as np

from .queue_worker import QueueWorker
from .audio_analyzer import AudioAnalyzer
from .preference_model import PreferenceModel
from .dj_analyzer import AIDJAnalyzer
from .segment_index import SegmentIndexManager

# Configure logging with file output
LOG_DIR = Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# File handler with rotation (10MB max, keep 5 backups)
file_handler = RotatingFileHandler(
    LOG_DIR / 'ml_worker.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Error-only file handler
error_handler = RotatingFileHandler(
    LOG_DIR / 'ml_worker_errors.log',
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3,
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler, error_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {LOG_DIR / 'ml_worker.log'}")

MIN_LIKES_FOR_TRAINING = 3


def trigger_auto_training_if_needed(worker: QueueWorker) -> bool:
    """Check if auto-training should be triggered and do it."""
    with worker.db_conn.cursor() as cursor:
        # Count liked tracks with embeddings
        cursor.execute("""
            SELECT COUNT(*) FROM tracks 
            WHERE is_liked = true AND embedding_path IS NOT NULL
        """)
        liked_count = cursor.fetchone()[0]
        
        if liked_count < MIN_LIKES_FOR_TRAINING:
            return False
        
        # Check if already training
        cursor.execute("SELECT status FROM preference_model_status LIMIT 1")
        row = cursor.fetchone()
        if row and row[0] == 'training':
            return False
        
        # Get liked track IDs
        cursor.execute("SELECT id FROM tracks WHERE is_liked = true AND embedding_path IS NOT NULL")
        liked_track_ids = [r[0] for r in cursor.fetchall()]
        
        cursor.execute("SELECT id FROM fragments WHERE is_liked = true AND embedding_path IS NOT NULL")
        liked_fragment_ids = [r[0] for r in cursor.fetchall()]
    
    logger.info(f"Auto-training triggered with {len(liked_track_ids)} liked tracks")
    
    # Train directly (we're already in the worker)
    task_data = {
        'liked_track_ids': liked_track_ids,
        'liked_fragment_ids': liked_fragment_ids,
    }
    
    try:
        handle_model_training(task_data, worker)
        return True
    except Exception as e:
        logger.error(f"Auto-training failed: {e}")
        return False


def handle_track_analysis(task_data: dict, worker: QueueWorker, analyzer: AudioAnalyzer) -> dict:
    """Handle track analysis task."""
    track_id = task_data['track_id']
    audio_path = task_data['audio_path']
    
    logger.info(f"Analyzing track {track_id}: {audio_path}")
    
    # Check if audio file exists - fail fast if not
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        worker.update_track_status(track_id, 'failed')
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check if track exists in database
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT id FROM tracks WHERE id = %s", (track_id,))
        if not cursor.fetchone():
            logger.error(f"Track {track_id} not found in database")
            raise ValueError(f"Track {track_id} not found in database")
    
    # Update track status to processing (track was in 'pending' until now)
    worker.update_track_status(track_id, 'processing')
    
    # Perform full analysis
    analysis = analyzer.analyze_full(audio_path)
    
    # Save embedding
    embedding_dir = Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'embeddings'
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    embedding_path = embedding_dir / f"track_{track_id}.npy"
    np.save(str(embedding_path), analysis['embedding'])
    
    # Update track in database
    analysis['embedding_path'] = str(embedding_path)
    worker.update_track_analysis(track_id, analysis)
    
    logger.info(f"Track {track_id} analysis complete")
    
    # Note: Auto-training is disabled here to avoid overhead during batch imports.
    # Training is triggered manually or via scheduled task.
    # trigger_auto_training_if_needed(worker)
    
    return analysis


def handle_fragment_analysis(task_data: dict, worker: QueueWorker, analyzer: AudioAnalyzer, segment_index: SegmentIndexManager) -> dict:
    """Handle fragment analysis task - extracts embedding from specific segment."""
    fragment_id = task_data['fragment_id']
    track_id = task_data['track_id']
    audio_path = task_data['audio_path']
    start_time = task_data['start_time']
    end_time = task_data['end_time']
    
    logger.info(f"Analyzing fragment {fragment_id} ({start_time}-{end_time}s)")
    
    # Check if audio file exists - fail fast if not
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found for fragment {fragment_id}: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Extract embedding for the SPECIFIC SEGMENT (not whole track)
    embedding = analyzer.extract_embeddings(audio_path, start_time=start_time, end_time=end_time)
    
    # Save embedding
    embedding_dir = Path(os.getenv('STORAGE_PATH', '/app/storage')) / 'embeddings'
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    embedding_path = embedding_dir / f"fragment_{fragment_id}.npy"
    np.save(str(embedding_path), embedding)
    
    # Update fragment in database
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE fragments 
            SET embedding_path = %s, analysis_status = 'completed'
            WHERE id = %s
            """,
            (str(embedding_path), fragment_id)
        )
    
    # Add to segment index for similarity search (Requirement 2.2)
    segment_index.add_segment(fragment_id, track_id, embedding, start_time, end_time)
    segment_index.save()
    
    logger.info(f"Fragment {fragment_id} analysis complete (segment {start_time:.1f}s - {end_time:.1f}s)")
    return {'embedding_path': str(embedding_path)}


def analyze_comment_sentiment(comment: str) -> dict:
    """
    Simple sentiment analysis for fragment comments.
    Returns sentiment features that can boost/reduce preference weight.
    """
    if not comment:
        return {'weight': 1.0, 'sentiment': 'neutral'}
    
    comment_lower = comment.lower()
    
    # Positive keywords (boost weight)
    positive_words = [
        # EN general positive
        'love', 'loved', 'lovely', 'amazing', 'awesome', 'great', 'best', 'favorite',
        'perfect', 'beautiful', 'incredible', 'fantastic', 'excellent', 'insane',
        'legendary', 'masterpiece', 'goated', 'goat', 'banger', 'slaps', 'slap',
        'hits', 'hits hard', 'fire', 'lit', 'dope', 'sick', 'crazy', 'wild',
        'fantabulous', 'stellar', 'brilliant', 'top tier', '10/10', '11/10',
        'on repeat', 'repeat', 'replay', 'relistening',

        # RU general positive
        'люблю', 'любимка', 'обожаю', 'обожаюсь', 'кайф', 'кайфую', 'кайфово',
        'круто', 'крутяк', 'крутейше', 'крутейший', 'класс', 'классно', 'супер',
        'шикарно', 'великолепно', 'прекрасно', 'красиво', 'бомба', 'имба', 'имбово',
        'топ', 'топчик', 'топово', 'лучший', 'лучшая', 'лучшее', 'пушка', 'пушечно',
        'огонь', 'огненно', 'разъёб', 'разнос', 'разносит', 'разрыв', 'разрывной',
        'сочно', 'жирно', 'мощно', 'сильно', 'сильнейше', 'чётко', 'чётенько',
        'потрясающе', 'офигенно', 'офигеть', 'офигенно!', 'охрененно', 'охренеть',
        'зашло', 'зашёл', 'зашла', 'вкатило', 'вкатывает', 'впёрло', 'прёт', 'прет',
        'тащит', 'тащусь', 'цепляет', 'залип', 'залипательно', 'залипаю',
        'шедевр', 'годно', 'годнота', 'кайфец', 'приятно', 'атмосферно', 'лампово',
        'ностальгия', 'ностальгично', 'вдохновляет', 'мурашки', 'мурашит', 'до мурашек',

        # RU profanity-positive (мат/жаргон)
        'ахуенно', 'охуенно', 'пиздато', 'заебись', 'заебато', 'ебать круто',
        'пиздец как круто', 'разъебало', 'разъёбало', 'охуеть как', 'ахуеть как',
        'ебануться', 'ебнуться', 'ебать', 'сука кайф', 'сука круто',

        # Music-specific EN
        'drop', 'bass', 'subbass', 'melody', 'hook', 'chorus', 'verse', 'bridge',
        'vocal', 'vocals', 'flow', 'groove', 'rhythm', 'beat', 'buildup', 'build-up',
        'breakdown', 'synth', 'lead', 'kick', 'snare', 'hi-hat', '808', 'mix', 'master',
        'mastering', 'production', 'arrangement', 'sound design', 'sounddesign',
        'clean mix', 'crispy', 'tight', 'punchy', 'wide', 'warm', 'lush', 'atmospheric',

        # Music-specific RU
        'бит', 'биток', 'кач', 'качает', 'качает жёстко', 'качает жестко', 'раскачало',
        'вибра', 'вибрация', 'виб', 'вайб', 'вайбово', 'атмосфера', 'атмосферка',
        'припев', 'куплет', 'хуки', 'хук', 'мелодия', 'гармония', 'вокал', 'голос',
        'флоу', 'читка', 'панч', 'панчи', 'рифмы', 'рифмует', 'саунд', 'саунд крутой',
        'сведение', 'мастеринг', 'продакшн', 'аранжировка', 'саунддизайн',
        'дроп', 'дропнуло', 'дропчик', 'бас', 'басуха', 'низ', 'низы', 'саб', 'саббас',
        'плотно', 'плотняк', 'чисто', 'чистый звук', 'широко', 'объёмно', 'объемно',
        'сочно звучит', 'вкусно', 'вкусный звук',

        # Actions that imply positive preference
        'в плейлист', 'в избранное', 'в любимое', 'добавил', 'добавила', 'сохранил',
        'сохранила', 'сохраню', 'сохранить', 'на репит', 'на повтор', 'на повторе',
        'переслушиваю', 'переслушал', 'переслушала', 'слушаю снова', 'слушаю ещё раз',
        'буду слушать', 'буду переслушивать',

        # Emojis / symbols (positive)
        '❤️', '❤', '💙', '💚', '💜', '🖤', '💖', '💗', '💘', '💝', '😍', '🥰', '🤩',
        '🔥', '💯', '✅', '⭐', '🌟', '✨', '👏', '🙌', '😻', '🎧', '🎶', '🔊'
    ]

    # Negative keywords (reduce weight)
    negative_words = [
        # EN general negative
        'bad', 'wack', 'weak', 'boring', 'skip', 'skipping', 'hate', 'hated',
        'worst', 'annoying', 'meh', 'mid', 'trash', 'garbage', 'awful', 'terrible',
        'cringe', 'corny', 'lame', 'overrated', 'forgettable', 'generic',
        'disappointing', 'painful', 'noise', 'noisy',

        # RU general negative
        'плохо', 'ужасно', 'отстой', 'отстойно', 'дичь', 'треш', 'трэш', 'мусор',
        'скучно', 'нудно', 'занудно', 'уныло', 'пресно', 'никак', 'не цепляет',
        'не зашло', 'не зашел', 'не зашла', 'не вкатило', 'не прёт', 'не прет',
        'надоело', 'надоедает', 'заебало', 'приелось', 'бесит', 'раздражает',
        'кринж', 'кринжово', 'стыд', 'стыдно', 'испанский стыд',
        'слабо', 'слабенько', 'посредственно', 'посредняк', 'так себе', 'такое себе',
        'не интересно', 'неинтересно', 'не нравится', 'не понравилось',
        'пропустить', 'скип', 'скипнуть', 'пролистал', 'пролистала',
        'глупо', 'тупо', 'тупняк', 'бред', 'ерунда', 'фигня', 'херня', 'лажа',
        'туфта', 'шляпа', 'шлак', 'параша', 'дно', 'днище', 'ни о чём', 'ни о чем',

        # RU profanity-negative (мат/жаргон)
        'хуйня', 'хуита', 'хуитень', 'ебанина', 'ебень', 'говно', 'говнище',
        'дерьмо', 'пиздец', 'пиздец плохо', 'пиздец ужас', 'сука бесит',
        'ебать как плохо', 'нахуй', 'иди нахуй', 'заебало', 'заебала', 'заебали',

        # Music-specific EN negative
        'offbeat', 'out of tune', 'clipping', 'distorted', 'muddy', 'flat',
        'thin', 'harsh', 'piercing', 'bad mix', 'bad mastering', 'overcompressed',
        'too loud', 'too quiet', 'no dynamics',

        # Music-specific RU negative
        'не качает', 'не качнуло', 'не раскачало', 'пусто', 'пустоват', 'пустовато',
        'грязно', 'грязный звук', 'каша', 'засрано', 'перегруз', 'перегружено',
        'плоско', 'тонко', 'режет уши', 'сверлит', 'шум', 'шумно',
        'бит мимо', 'вокал мимо', 'фальш', 'фальшиво', 'не в тон', 'не строит',
        'плохое сведение', 'плохой микс', 'плохой мастер', 'перекомпресс',
        'однообразно', 'монотонно', 'затянуто', 'затянули', 'слишком долго',
        'нет дропа', 'дроп слабый', 'без дропа', 'нет хука', 'хука нет',

        # Actions that imply negative preference
        'удалил', 'удалила', 'не сохраню', 'не добавлю', 'не в плейлист',
        'пропускаю', 'пропустил', 'пропустила', 'скипаю', 'скипнул', 'скипнула',

        # Emojis / symbols (negative)
        '💩', '🤮', '🤢', '🙄', '😒', '😴', '🤦', '🤦‍♂️', '🤦‍♀️', '❌', '⛔', '🚫'
    ]
    
    positive_count = sum(1 for word in positive_words if word in comment_lower)
    negative_count = sum(1 for word in negative_words if word in comment_lower)
    
    # Calculate weight multiplier (0.5 to 2.0)
    if positive_count > negative_count:
        weight = min(2.0, 1.0 + positive_count * 0.2)
        sentiment = 'positive'
    elif negative_count > positive_count:
        weight = max(0.5, 1.0 - negative_count * 0.2)
        sentiment = 'negative'
    else:
        weight = 1.0
        sentiment = 'neutral'
    
    return {'weight': weight, 'sentiment': sentiment}


def handle_model_training(task_data: dict, worker: QueueWorker) -> dict:
    """Handle preference model training task with fragment and comment support."""
    liked_track_ids = task_data.get('liked_track_ids', [])
    liked_fragment_ids = task_data.get('liked_fragment_ids', [])
    
    logger.info(f"Training model with {len(liked_track_ids)} liked tracks, {len(liked_fragment_ids)} liked fragments")
    
    # Load embeddings for liked items
    storage_path = Path(os.getenv('STORAGE_PATH', '/app/storage'))
    embeddings_dir = storage_path / 'embeddings'
    
    liked_embeddings = []
    liked_weights = []  # Weight based on rating and comment sentiment
    all_embeddings = []
    
    # Load track embeddings
    for track_id in liked_track_ids:
        embedding_path = embeddings_dir / f"track_{track_id}.npy"
        if embedding_path.exists():
            emb = np.load(str(embedding_path))
            liked_embeddings.append(emb)
            liked_weights.append(1.0)  # Default weight for liked tracks
            all_embeddings.append(emb)
    
    # Load fragment embeddings with comment analysis
    with worker.db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT f.id, f.embedding_path, f.comment, f.track_id
            FROM fragments f
            WHERE f.is_liked = true AND f.embedding_path IS NOT NULL
        """)
        fragment_rows = cursor.fetchall()
        
        for row in fragment_rows:
            frag_id, emb_path, comment, track_id = row
            if emb_path and Path(emb_path).exists():
                emb = np.load(emb_path)
                liked_embeddings.append(emb)
                
                # Base weight for liked fragments
                base_weight = 1.0
                
                # Analyze comment sentiment
                comment_analysis = analyze_comment_sentiment(comment)
                final_weight = base_weight * comment_analysis['weight']
                
                liked_weights.append(final_weight)
                all_embeddings.append(emb)
                
                if comment:
                    logger.info(f"Fragment {frag_id}: comment sentiment={comment_analysis['sentiment']}, weight={final_weight:.2f}")
    
    # Load all track embeddings for negative sampling
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT id, embedding_path FROM tracks WHERE embedding_path IS NOT NULL")
        rows = cursor.fetchall()
        for row in rows:
            track_id, emb_path = row
            if emb_path and Path(emb_path).exists():
                emb = np.load(emb_path)
                if track_id not in liked_track_ids:
                    all_embeddings.append(emb)
    
    if len(liked_embeddings) == 0:
        raise ValueError("No embeddings found for liked items")
    
    liked_embeddings = np.array(liked_embeddings)
    liked_weights = np.array(liked_weights)
    all_embeddings = np.array(all_embeddings)
    
    logger.info(f"Loaded {len(liked_embeddings)} liked embeddings (with weights), {len(all_embeddings)} total embeddings")
    
    # Train model with weighted samples
    model = PreferenceModel(embedding_dim=liked_embeddings.shape[1])
    history = model.train(liked_embeddings, all_embeddings, epochs=100, sample_weights=liked_weights)
    
    # Save model
    models_dir = Path(os.getenv('MODELS_PATH', '/app/models'))
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'preference_model.npz'
    model.save(str(model_path))
    
    # Update preference_model_status in database
    with worker.db_conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE preference_model_status 
            SET status = 'trained', model_path = %s, trained_at = %s, likes_count = %s
            WHERE id = (SELECT id FROM preference_model_status LIMIT 1)
            """,
            (str(model_path), datetime.now(), len(liked_track_ids) + len(liked_fragment_ids))
        )
    
    logger.info(f"Model training complete, saved to {model_path}")
    return {'model_path': str(model_path), 'final_loss': history['loss'][-1] if history['loss'] else 0}


def handle_dj_analysis(task_data: dict, worker: QueueWorker, dj_analyzer: AIDJAnalyzer) -> dict:
    """
    Handle DJ analysis task.
    
    Analyzes track for DJ features: Camelot key, BPM, energy, structural points.
    Saves results to track_dj_analysis table.
    
    Requirements: 27.5, 27.6
    """
    track_id = task_data['track_id']
    audio_path = task_data['audio_path']
    
    logger.info(f"DJ analyzing track {track_id}: {audio_path}")
    
    # Perform DJ analysis
    result = dj_analyzer.analyze(audio_path, track_id)
    
    # Save to database
    with worker.db_conn.cursor() as cursor:
        # Check if analysis already exists
        cursor.execute(
            "SELECT id FROM track_dj_analysis WHERE track_id = %s",
            (track_id,)
        )
        existing = cursor.fetchone()
        
        import json
        drop_points_json = json.dumps(result.drop_points)
        breakdown_points_json = json.dumps(result.breakdown_points)
        
        if existing:
            # Update existing
            cursor.execute(
                """
                UPDATE track_dj_analysis 
                SET camelot_key = %s, bpm = %s, energy_level = %s,
                    intro_point = %s, outro_point = %s,
                    drop_points = %s, breakdown_points = %s,
                    analyzed_at = %s
                WHERE track_id = %s
                """,
                (
                    result.camelot_key,
                    result.bpm,
                    result.energy_level,
                    result.intro_point,
                    result.outro_point,
                    drop_points_json,
                    breakdown_points_json,
                    datetime.now(),
                    track_id
                )
            )
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO track_dj_analysis 
                (track_id, camelot_key, bpm, energy_level, intro_point, outro_point, 
                 drop_points, breakdown_points, analyzed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    track_id,
                    result.camelot_key,
                    result.bpm,
                    result.energy_level,
                    result.intro_point,
                    result.outro_point,
                    drop_points_json,
                    breakdown_points_json,
                    datetime.now()
                )
            )
    
    logger.info(f"DJ analysis complete for track {track_id}: {result.camelot_key}, {result.bpm} BPM")
    
    return {
        'track_id': track_id,
        'camelot_key': result.camelot_key,
        'bpm': result.bpm,
        'energy_level': result.energy_level,
    }


def handle_segment_search(
    task_data: dict,
    worker: QueueWorker,
    analyzer: AudioAnalyzer,
    segment_index: SegmentIndexManager
) -> dict:
    """
    Handle segment similarity search task.
    
    Extracts embedding from specified time range and searches for similar segments.
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    track_id = task_data['track_id']
    start_time = task_data['start_time']
    end_time = task_data['end_time']
    limit = task_data.get('limit', 10)
    min_similarity = task_data.get('min_similarity', 0.3)
    
    logger.info(f"Segment search: track {track_id}, {start_time:.1f}s - {end_time:.1f}s")
    
    # Get track audio path from database
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT file_path FROM tracks WHERE id = %s", (track_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Track {track_id} not found")
        file_path = row[0]
    
    audio_path = f"/app/storage/{file_path}"
    
    # Extract embedding for the query segment
    query_embedding = analyzer.extract_embeddings(audio_path, start_time=start_time, end_time=end_time)
    
    # Search for similar segments
    matches = segment_index.search_similar(
        query_embedding,
        k=limit,
        exclude_track_id=track_id,
        min_similarity=min_similarity
    )
    
    # Enrich results with track info
    results = []
    for match in matches:
        with worker.db_conn.cursor() as cursor:
            cursor.execute(
                "SELECT title, artist FROM tracks WHERE id = %s",
                (match.track_id,)
            )
            track_row = cursor.fetchone()
            if track_row:
                results.append({
                    'fragment_id': match.fragment_id,
                    'track_id': match.track_id,
                    'track_title': track_row[0],
                    'artist': track_row[1],
                    'start_time': match.start_time,
                    'end_time': match.end_time,
                    'similarity': match.similarity
                })
    
    logger.info(f"Segment search found {len(results)} matches")
    
    return {
        'query': {
            'track_id': track_id,
            'start_time': start_time,
            'end_time': end_time
        },
        'results': results,
        'total': len(results)
    }


def handle_embedding_recommendations(task_data: dict, worker: QueueWorker) -> dict:
    """
    Handle personalized recommendations based on embeddings.
    
    Uses cosine similarity between liked track embeddings and candidate tracks
    to find the most similar unheard tracks.
    
    Returns recommendations with explanations based on which liked tracks
    they are most similar to.
    """
    import redis
    
    limit = task_data.get('limit', 5)
    task_id = task_data.get('task_id')
    
    logger.info(f"Generating embedding-based recommendations (limit={limit})")
    
    storage_path = Path(os.getenv('STORAGE_PATH', '/app/storage'))
    embeddings_dir = storage_path / 'embeddings'
    
    # Get liked tracks with embeddings
    with worker.db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.embedding_path
            FROM tracks t
            WHERE t.is_liked = true AND t.embedding_path IS NOT NULL
        """)
        liked_tracks = cursor.fetchall()
        
        # Also get highly rated tracks (4-5 stars)
        cursor.execute("""
            SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.embedding_path
            FROM tracks t
            JOIN track_ratings tr ON t.id = tr.track_id
            WHERE tr.rating >= 4 AND t.embedding_path IS NOT NULL
        """)
        rated_tracks = cursor.fetchall()
    
    # Combine and deduplicate
    liked_map = {}
    for row in liked_tracks + rated_tracks:
        track_id, title, artist, genre, bpm, mood, emb_path = row
        if track_id not in liked_map:
            liked_map[track_id] = {
                'id': track_id,
                'title': title,
                'artist': artist,
                'genre': genre,
                'bpm': bpm,
                'mood': mood,
                'embedding_path': emb_path
            }
    
    if not liked_map:
        result = {
            'recommendations': [],
            'has_preferences': False,
            'message': 'Недостаточно данных. Поставьте лайки или оценки трекам.'
        }
        _store_result(task_id, result)
        return result
    
    # Load liked embeddings
    liked_embeddings = []
    liked_info = []
    for track_id, info in liked_map.items():
        emb_path = info['embedding_path']
        if emb_path and Path(emb_path).exists():
            emb = np.load(emb_path)
            liked_embeddings.append(emb)
            liked_info.append(info)
    
    if not liked_embeddings:
        result = {
            'recommendations': [],
            'has_preferences': False,
            'message': 'Не найдены embeddings для лайкнутых треков.'
        }
        _store_result(task_id, result)
        return result
    
    liked_embeddings = np.array(liked_embeddings)
    
    # Get listening history to exclude
    with worker.db_conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT track_id FROM listening_history")
        listened_ids = set(row[0] for row in cursor.fetchall())
    
    # Get candidate tracks (not liked, not listened, with embeddings)
    liked_ids = set(liked_map.keys())
    with worker.db_conn.cursor() as cursor:
        cursor.execute("""
            SELECT t.id, t.title, t.artist, t.genre, t.bpm, t.mood, t.key, t.embedding_path
            FROM tracks t
            WHERE t.is_liked = false 
              AND t.analysis_status = 'completed'
              AND t.embedding_path IS NOT NULL
        """)
        candidates = cursor.fetchall()
    
    # Filter out listened tracks
    candidates = [c for c in candidates if c[0] not in listened_ids and c[0] not in liked_ids]
    
    if not candidates:
        result = {
            'recommendations': [],
            'has_preferences': True,
            'message': 'Вы уже прослушали все доступные треки!'
        }
        _store_result(task_id, result)
        return result
    
    # Calculate similarity scores for each candidate
    recommendations = []
    
    for candidate in candidates:
        track_id, title, artist, genre, bpm, mood, key, emb_path = candidate
        
        if not emb_path or not Path(emb_path).exists():
            continue
        
        candidate_emb = np.load(emb_path)
        
        # Calculate cosine similarity with all liked tracks
        similarities = []
        for i, liked_emb in enumerate(liked_embeddings):
            # Cosine similarity
            sim = np.dot(candidate_emb, liked_emb) / (np.linalg.norm(candidate_emb) * np.linalg.norm(liked_emb) + 1e-10)
            similarities.append((sim, liked_info[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Get top 3 most similar liked tracks for explanation
        top_similar = similarities[:3]
        avg_similarity = np.mean([s[0] for s in top_similar])
        
        # Build explanation
        reasons = []
        for sim, liked_track in top_similar:
            if sim > 0.7:
                reasons.append(f"Очень похож на \"{liked_track['title']}\" ({sim*100:.0f}%)")
            elif sim > 0.5:
                reasons.append(f"Похож на \"{liked_track['title']}\" ({sim*100:.0f}%)")
        
        # Add genre/mood context if matches
        top_genres = [info['genre'] for info in liked_info if info['genre']]
        top_moods = [info['mood'] for info in liked_info if info['mood']]
        
        if genre and genre in top_genres:
            reasons.append(f"Жанр {genre} — ваш любимый")
        if mood and mood in top_moods:
            reasons.append(f"Настроение \"{mood}\" вам подходит")
        
        recommendations.append({
            'track': {
                'id': track_id,
                'title': title,
                'artist': artist,
                'genre': genre,
                'bpm': bpm,
                'mood': mood,
                'key': key
            },
            'score': float(avg_similarity),
            'reasons': reasons[:4],  # Max 4 reasons
            'explanation': '. '.join(reasons[:3]) if reasons else 'Может вам понравиться'
        })
    
    # Sort by score and take top N
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    recommendations = recommendations[:limit]
    
    # Build preference summary
    preference_summary = {
        'top_genres': list(set(info['genre'] for info in liked_info if info['genre']))[:3],
        'top_moods': list(set(info['mood'] for info in liked_info if info['mood']))[:2],
        'avg_bpm': int(np.mean([info['bpm'] for info in liked_info if info['bpm']]) or 140),
        'tracks_analyzed': len(liked_info)
    }
    
    result = {
        'recommendations': recommendations,
        'total': len(recommendations),
        'preference_summary': preference_summary,
        'has_preferences': True,
        'message': None if recommendations else 'Не нашёл подходящих треков.'
    }
    
    logger.info(f"Generated {len(recommendations)} embedding-based recommendations")
    
    # Store result in Redis for PHP to retrieve
    _store_result(task_id, result)
    
    return result


def _store_result(task_id: str, result: dict) -> None:
    """Store result in Redis for retrieval by PHP."""
    if not task_id:
        return
    
    import redis
    import json
    
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD') or None,
        decode_responses=True
    )
    
    result_key = f"chat_recommend_result:{task_id}"
    redis_client.setex(result_key, 60, json.dumps(result))  # 60 second TTL
    logger.info(f"Stored result in Redis: {result_key}")


def main():
    """Main entry point."""
    logger.info("Starting ML Worker...")
    
    # Initialize components
    worker = QueueWorker()
    analyzer = AudioAnalyzer()
    dj_analyzer = AIDJAnalyzer()
    segment_index = SegmentIndexManager()
    
    # Register handlers
    worker.register_handler(
        'track_analysis',
        lambda task: handle_track_analysis(task, worker, analyzer)
    )
    worker.register_handler(
        'fragment_analysis',
        lambda task: handle_fragment_analysis(task, worker, analyzer, segment_index)
    )
    worker.register_handler(
        'model_training',
        lambda task: handle_model_training(task, worker)
    )
    worker.register_handler(
        'dj_analysis',
        lambda task: handle_dj_analysis(task, worker, dj_analyzer)
    )
    worker.register_handler(
        'segment_search',
        lambda task: handle_segment_search(task, worker, analyzer, segment_index)
    )
    worker.register_handler(
        'embedding_recommendations',
        lambda task: handle_embedding_recommendations(task, worker)
    )
    
    # Handle shutdown signals
    def shutdown(signum, frame):
        logger.info("Received shutdown signal")
        worker.stop()
        
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    # Start worker
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("ML Worker stopped")


if __name__ == '__main__':
    main()
