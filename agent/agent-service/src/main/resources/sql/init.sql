-- Create database (if not exists)
CREATE DATABASE IF NOT EXISTS agent_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Use database
USE agent_db;

-- Create dataset table
CREATE TABLE IF NOT EXISTS dataset (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT 'Dataset ID',
    name VARCHAR(255) NOT NULL COMMENT 'Dataset name',
    description TEXT COMMENT 'Dataset description',
    upload_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Upload time',
    status VARCHAR(20) NOT NULL COMMENT 'Status (preprocessed, not preprocessed)',
    dataset_path VARCHAR(512) NOT NULL COMMENT 'Dataset absolute path',
    processed_path VARCHAR(512) COMMENT 'Preprocessed dataset absolute path',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Create time',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create task table
CREATE TABLE IF NOT EXISTS task (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT 'Task ID',
    description TEXT COMMENT 'Task description',
    start_time DATETIME COMMENT 'Start time',
    complete_time DATETIME COMMENT 'Complete time',
    dataset_id BIGINT NOT NULL COMMENT 'Dataset ID',
    model_weight_path VARCHAR(512) COMMENT 'Model weight absolute path',
    test_result_path VARCHAR(512) COMMENT 'Test result absolute path',
    result TEXT COMMENT 'Task result JSON',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Create time',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    FOREIGN KEY (dataset_id) REFERENCES dataset(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
