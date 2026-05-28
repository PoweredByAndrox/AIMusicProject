<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('friendships', function (Blueprint $table) {
            $table->id();
            $table->foreignId('requester_id')->constrained('users')->cascadeOnDelete();
            $table->foreignId('addressee_id')->constrained('users')->cascadeOnDelete();
            $table->string('status', 24)->default('pending');
            $table->timestamp('responded_at')->nullable();
            $table->timestamp('accepted_at')->nullable();
            $table->timestamps();

            $table->unique(['requester_id', 'addressee_id'], 'friendships_requester_addressee_unique');
            $table->index(['requester_id', 'status'], 'idx_friendships_requester_status');
            $table->index(['addressee_id', 'status'], 'idx_friendships_addressee_status');
        });

        Schema::create('direct_messages', function (Blueprint $table) {
            $table->id();
            $table->foreignId('sender_id')->constrained('users')->cascadeOnDelete();
            $table->foreignId('recipient_id')->constrained('users')->cascadeOnDelete();
            $table->string('type', 32)->default('message');
            $table->text('body')->nullable();
            $table->jsonb('payload')->nullable();
            $table->timestamp('read_at')->nullable();
            $table->timestamps();

            $table->index(['recipient_id', 'read_at'], 'idx_direct_messages_recipient_read_at');
            $table->index(['sender_id', 'recipient_id', 'created_at'], 'idx_direct_messages_sender_recipient_created_at');
            $table->index(['recipient_id', 'sender_id', 'created_at'], 'idx_direct_messages_recipient_sender_created_at');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('direct_messages');
        Schema::dropIfExists('friendships');
    }
};
