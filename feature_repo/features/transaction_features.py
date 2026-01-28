from datetime import timedelta
from feast import Entity, FeatureView, Field, ValueType
from feast.types import Float32, Int64, String
from feast.infra.offline_stores.file_source import FileSource

# Define entities
user = Entity(name="user_id", value_type=ValueType.STRING)
merchant = Entity(name="merchant_id", value_type=ValueType.STRING)

# Define data sources
transaction_stats_source = FileSource(
    path="data/processed/transaction_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

user_behavior_source = FileSource(
    path="data/processed/user_behavior.parquet",
    timestamp_field="event_timestamp"
)

# Define feature views
transaction_features = FeatureView(
    name="transaction_features",
    entities=[user, merchant],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="avg_transaction_amount_24h", dtype=Float32),
        Field(name="transaction_count_24h", dtype=Int64),
        Field(name="fraud_count_24h", dtype=Int64),
        Field(name="max_transaction_amount_24h", dtype=Float32),
        Field(name="last_transaction_hours_ago", dtype=Float32),
    ],
    source=transaction_stats_source
)

user_behavior_features = FeatureView(
    name="user_behavior_features",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="user_trust_score", dtype=Float32),
        Field(name="avg_daily_transactions", dtype=Float32),
        Field(name="preferred_merchant_categories", dtype=String),
        Field(name="typical_transaction_hours", dtype=String),
        Field(name="device_count", dtype=Int64),
    ],
    source=user_behavior_source
)
