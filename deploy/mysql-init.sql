-- Runs only when the official MySQL image initializes a NEW project volume.
-- Never run this against the native 3306 instance; existing volumes are not migrated here.
USE `fault_diagnosis`;

-- Keep this schema aligned with qa_platform.storage.sql_conversations.Base.
-- Defaults/timestamps are supplied by the application, matching the SQLAlchemy mappings.
CREATE TABLE `qa_conversations` (
    `id` VARCHAR(32) NOT NULL,
    `created_at` DATETIME NOT NULL,
    `last_turn` INTEGER NOT NULL,
    `active` BOOL NOT NULL,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB;

CREATE TABLE `qa_turns` (
    `conversation_id` VARCHAR(32) NOT NULL,
    `turn_no` INTEGER NOT NULL,
    `query` TEXT NOT NULL,
    `effective_query` TEXT NOT NULL,
    `model` VARCHAR(80) NOT NULL,
    `summary` JSON NOT NULL,
    `result` JSON NOT NULL,
    `created_at` DATETIME NOT NULL,
    PRIMARY KEY (`conversation_id`, `turn_no`),
    FOREIGN KEY (`conversation_id`) REFERENCES `qa_conversations` (`id`)
) ENGINE=InnoDB;

-- The official image initially grants this account schema-wide ALL; narrow it after table setup.
-- This only changes the container-created account, not any native MySQL account.
REVOKE ALL PRIVILEGES ON `fault\_diagnosis`.* FROM 'fault_qa'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON `fault\_diagnosis`.* TO 'fault_qa'@'%';
