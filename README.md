1 users

id (PK)

email, phone, full_name

password_hash

role (ADMIN, USER)

device_id (FK → devices.id) one device per user

points balance

level_id (FK → levels.id)

created at, updated at

2 devices

id (PK)

device name, manufacturer, model

3 device_capability

id (PK)

device id (FK → devices.id)

capability_code (e.g., STEPS, HRM, SLEEP)

Note: A single device can have multiple capabilities

4 levels

id (PK)

level name

point threshold

5 tasks (evergreen, threshold-based)

id (PK)

name, description

required_level (FK → levels.id)

required metric (matches a capability_code; validated in app)

target_value (threshold to complete)

iCOOL SETIVE/INACTIVE)

6 challenges (time-bound, curated set of tasks)

id (PK)

name, description

required level (FK levels.id)

required_metric (minimum device capability; validated in app)

start time, end time

status (ACTIVE, EXPIRED, RANKED)

reward_scheme (RANK_BASED, NONE)

created_by_user_id (FK users.id, nullable; null/ADMIN implies curated by admin)

Notes: For user-created challenges, set reward_scheme NONE (no points payout).

7 challenge_task (curated task set per challenge)

id (PK)

challenge_id (FK → challenges.id)

task id (FK tasks.id)

UNIQUE (challenge_id, task_id)

8 user_task

id (PK)

user id (FK → users.id)

task id (FK tasks.id)

progress_value

status (IN PROGRESS, COMPLETED)

completed_at (nullable)

points awarded (lifetime points awarded for this task instance if single-completion design)

UNIQUE (user_id, task_id)

9 user_challenge (per-user participation and score, updated by batch)

id (PK)

user_id (FK users.id)

challenge_id (FK challenges.id)

status (JOINED, COMPLETED, RANKED)

progress_value (optional aggregate)

final_score (computed by batch as sum of task completion points within window and within the curated set)

rank (nullable until RANKED)

points_awarded (points from challenge payout; zero for user-created with reward_scheme = NONE)

ranked at (nullable)

UNIQUE(user_id, challenge_id)


UNIQUE (user_id, challenge_id)

10 leaderboard (persistent final ranking)

id (PK)

challenge_id (FK → challenges.id)

user_id (FK users.id)

final_score

rank

awarded points

generated_at

UNIQUE (challenge_id, user_id)

11 user_activity_event (Kafka-consumed raw activity)

id (PK)

event_id (business idempotency key) - UNIQUE

user_id (FK → users.id)

metric_type (e.g., STEPS, HRM, SLEEP)

metric_value

event_time

processed_status (PENDING, PROCESSED, DEAD_LETTER)

processed_at (nullable)

