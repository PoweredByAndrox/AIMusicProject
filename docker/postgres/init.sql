-- PostgreSQL initialization script
-- This file is executed when the PostgreSQL container is first created

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges (if needed for specific users)
-- GRANT ALL PRIVILEGES ON DATABASE music_recommendations TO postgres;

-- Note: Tables will be created by Laravel migrations
