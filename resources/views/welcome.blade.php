<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AI Music Recommendations</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            text-align: center;
            padding: 2rem;
        }
        h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #e94560, #0f3460);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        p {
            font-size: 1.2rem;
            color: #a0a0a0;
            margin-bottom: 2rem;
        }
        .status {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: rgba(233, 69, 96, 0.2);
            border: 1px solid #e94560;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .features {
            margin-top: 3rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            max-width: 800px;
        }
        .feature {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .feature h3 {
            color: #e94560;
            margin-bottom: 0.5rem;
        }
        .feature p {
            font-size: 0.9rem;
            margin: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 AI Music Recommendations</h1>
        <p>Personal music recommendation system powered by machine learning</p>
        <span class="status">Laravel {{ app()->version() }} • PHP {{ PHP_VERSION }}</span>
        
        <div class="features">
            <div class="feature">
                <h3>📤 Upload</h3>
                <p>Upload MP3/WAV files or import from YouTube</p>
            </div>
            <div class="feature">
                <h3>🔬 Analyze</h3>
                <p>ML-powered genre, BPM, mood detection</p>
            </div>
            <div class="feature">
                <h3>❤️ Like</h3>
                <p>Mark favorite tracks and fragments</p>
            </div>
            <div class="feature">
                <h3>🎯 Recommend</h3>
                <p>Get personalized recommendations</p>
            </div>
        </div>
    </div>
</body>
</html>
