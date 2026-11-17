"""Initial schema migration for PostgreSQL and SQLite.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-08

Defines all 11 audited relational entities:
- users
- sessions (user_id unconstrained for anonymous device support)
- messages
- conversation_logs
- user_memories (user_id unconstrained for anonymous memory support)
- projects
- library_files
- image_records
- scheduled_tasks
- task_executions (task_id unconstrained for legacy test executions support)
- user_integrations
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Users
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('username', sa.String(length=128), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=64), server_default='student', nullable=True),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('salt', sa.String(length=128), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_users_email', 'users', ['email'], unique=True)
    op.create_index('idx_users_username', 'users', ['username'], unique=True)

    # 2. Sessions (user_id unconstrained for anonymous IDs and legacy NULL)
    op.create_table(
        'sessions',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=128), nullable=True),
        sa.Column('title', sa.String(length=255), server_default='New Conversation', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id'),
    )
    op.create_index('idx_sessions_session_id', 'sessions', ['session_id'])
    op.create_index('idx_sessions_user', 'sessions', ['user_id'])
    op.create_index('idx_sessions_updated', 'sessions', ['updated_at'])
    op.create_index('idx_sessions_user_updated', 'sessions', ['user_id', 'updated_at'])

    # 3. Messages
    op.create_table(
        'messages',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=128), nullable=False),
        sa.Column('role', sa.String(length=32), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_messages_session', 'messages', ['session_id'])
    op.create_index('idx_messages_session_id', 'messages', ['session_id', 'id'])

    # 4. Conversation Logs
    op.create_table(
        'conversation_logs',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('user_query', sa.Text(), nullable=False),
        sa.Column('retrieved_chunks', sa.Text(), server_default='[]', nullable=False),
        sa.Column('provider_used', sa.String(length=64), nullable=False),
        sa.Column('model_response', sa.Text(), nullable=False),
        sa.Column('response_time_ms', sa.Float(), nullable=False),
        sa.Column('feedback', sa.String(length=32), nullable=True),
        sa.Column('shadow_response', sa.Text(), nullable=True),
        sa.Column('shadow_model_id', sa.String(length=128), nullable=True),
        sa.Column('served_model_version', sa.String(length=64), nullable=True),
        sa.Column('ab_test_group', sa.String(length=64), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('cost_usd', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_conv_logs_timestamp', 'conversation_logs', ['timestamp'])

    # 5. User Memories (user_id unconstrained for anonymous memories)
    op.create_table(
        'user_memories',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=128), nullable=False),
        sa.Column('category', sa.String(length=64), server_default='general', nullable=False),
        sa.Column('fact', sa.Text(), nullable=False),
        sa.Column('source', sa.String(length=32), server_default='auto', nullable=True),
        sa.Column('confidence', sa.Float(), server_default='1.0', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_memories_user', 'user_memories', ['user_id'])
    op.create_index('idx_memories_category', 'user_memories', ['category'])
    op.create_index('idx_memories_updated', 'user_memories', ['updated_at'])
    op.create_index('idx_memories_user_cat', 'user_memories', ['user_id', 'category'])

    # 6. Projects
    op.create_table(
        'projects',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=128), server_default='guest', nullable=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('instructions', sa.Text(), nullable=True),
        sa.Column('preferred_model', sa.String(length=128), server_default='auto', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_projects_user', 'projects', ['user_id'])

    # 7. Library Files
    op.create_table(
        'library_files',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=128), server_default='guest', nullable=True),
        sa.Column('project_id', sa.String(length=128), nullable=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=64), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('storage_path', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), server_default='indexed', nullable=True),
        sa.Column('chunk_count', sa.Integer(), server_default='0', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_library_user', 'library_files', ['user_id'])
    op.create_index('idx_library_project', 'library_files', ['project_id'])

    # 8. Image Records
    op.create_table(
        'image_records',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=128), server_default='guest', nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('enhanced_prompt', sa.Text(), nullable=True),
        sa.Column('model', sa.String(length=64), nullable=False),
        sa.Column('tier', sa.String(length=16), server_default='B', nullable=True),
        sa.Column('style', sa.String(length=64), server_default='photorealistic', nullable=True),
        sa.Column('width', sa.Integer(), server_default='1024', nullable=True),
        sa.Column('height', sa.Integer(), server_default='1024', nullable=True),
        sa.Column('file_url', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_images_user', 'image_records', ['user_id', 'created_at'])

    # 9. Scheduled Tasks
    op.create_table(
        'scheduled_tasks',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=128), server_default='guest', nullable=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('cron_expression', sa.String(length=64), server_default='0 8 * * *', nullable=False),
        sa.Column('timezone', sa.String(length=64), server_default='UTC', nullable=True),
        sa.Column('status', sa.String(length=32), server_default='active', nullable=True),
        sa.Column('last_run', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_run', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_scheduled_user', 'scheduled_tasks', ['user_id'])

    # 10. Task Executions (task_id unconstrained for legacy test executions support)
    op.create_table(
        'task_executions',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('task_id', sa.String(length=128), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('output', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('executed_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_task_exec', 'task_executions', ['task_id', 'executed_at'])

    # 11. User Integrations
    op.create_table(
        'user_integrations',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=128), server_default='guest', nullable=True),
        sa.Column('plugin_id', sa.String(length=128), nullable=False),
        sa.Column('is_enabled', sa.Integer(), server_default='1', nullable=True),
        sa.Column('config_encrypted', sa.Text(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'plugin_id', name='uq_user_plugin'),
    )


def downgrade() -> None:
    op.drop_table('user_integrations')
    op.drop_table('task_executions')
    op.drop_table('scheduled_tasks')
    op.drop_table('image_records')
    op.drop_table('library_files')
    op.drop_table('projects')
    op.drop_table('user_memories')
    op.drop_table('conversation_logs')
    op.drop_table('messages')
    op.drop_table('sessions')
    op.drop_table('users')
