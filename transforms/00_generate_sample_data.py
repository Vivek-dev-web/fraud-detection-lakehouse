# Databricks notebook source
# MAGIC %md
# MAGIC ### Generate synthetic fraud-detection data
# MAGIC Lands synthetic accounts/merchants dimensions plus transactions into a Unity Catalog
# MAGIC volume: one **labeled historical batch** (ground-truth `is_fraud`, used for model
# MAGIC training) and several **unlabeled streaming batches** (`is_fraud` = null, the live
# MAGIC feed the pipeline ingests and later scores). Every transaction file shares the same
# MAGIC schema (including `is_fraud`) so Auto Loader never has to deal with schema evolution.
# MAGIC
# MAGIC Fraud is injected via simple rules (large foreign card-not-present transactions,
# MAGIC transaction velocity, amount outliers) plus label noise, so the resulting dataset is
# MAGIC learnable but not trivially rule-matchable -- good enough for a demo classifier.

# COMMAND ----------
# MAGIC %pip install faker==25.8.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "fraud_detection")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

RAW_ROOT = f"/Volumes/{catalog}/{schema}/raw_landing"

# COMMAND ----------
import json
import random
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()
Faker.seed(7)
random.seed(7)

COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "NG", "RU", "CN", "AU"]
HIGH_RISK_COUNTRIES = {"NG", "RU"}
CATEGORIES = ["grocery", "electronics", "travel", "gambling", "fuel", "restaurant", "jewelry", "online_retail"]


def write_json_lines(path, records):
    dbutils.fs.mkdirs(path.rsplit("/", 1)[0])
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# COMMAND ----------
# dimensions
num_accounts, num_merchants = 500, 80

accounts = []
for i in range(1, num_accounts + 1):
    home_country = random.choice(COUNTRIES)
    accounts.append(
        {
            "account_id": f"A{i:05d}",
            "customer_name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "home_country": home_country,
            "risk_segment": random.choices(["LOW", "MEDIUM", "HIGH"], weights=[0.7, 0.22, 0.08])[0],
            "account_open_date": fake.date_between(start_date="-5y", end_date="-90d").isoformat(),
        }
    )
write_json_lines(f"{RAW_ROOT}/accounts/accounts.json", accounts)

merchants = [
    {
        "merchant_id": f"M{i:04d}",
        "merchant_name": fake.company(),
        "category": random.choice(CATEGORIES),
        "country": random.choice(COUNTRIES),
    }
    for i in range(1, num_merchants + 1)
]
write_json_lines(f"{RAW_ROOT}/merchants/merchants.json", merchants)

# COMMAND ----------
def make_transaction(txn_id, account, ts, force_fraud_pattern=None):
    merchant = random.choice(merchants)
    is_present = random.random() > 0.35
    txn_country = merchant["country"] if random.random() > 0.15 else random.choice(COUNTRIES)
    amount = round(random.lognormvariate(3.2, 1.0), 2)

    if force_fraud_pattern == "large_foreign_cnp":
        amount = round(random.uniform(800, 5000), 2)
        txn_country = random.choice(list(HIGH_RISK_COUNTRIES))
        is_present = False
    elif force_fraud_pattern == "amount_outlier":
        amount = round(random.uniform(2000, 9000), 2)

    return {
        "transaction_id": txn_id,
        "account_id": account["account_id"],
        "merchant_id": merchant["merchant_id"],
        "amount": amount,
        "currency": "USD",
        "transaction_ts": ts.isoformat(),
        "is_card_present": is_present,
        "device_id": f"DEV{random.randint(1, num_accounts * 2):06d}",
        "txn_country": txn_country,
    }


def label_is_fraud(txn, account, velocity_count, noise_roll):
    foreign = txn["txn_country"] != account["home_country"]
    high_risk_country = txn["txn_country"] in HIGH_RISK_COUNTRIES
    rule_hit = (
        (txn["amount"] > 800 and not txn["is_card_present"] and (foreign or high_risk_country))
        or (txn["amount"] > 2000)
        or (velocity_count >= 5)
    )
    if rule_hit:
        return noise_roll > 0.08  # 8% label noise on true positives
    return noise_roll < 0.01  # 1% random false positives (mislabeled history)


# COMMAND ----------
# labeled historical batch (for training)
base_day = datetime(2026, 5, 1)
labeled = []
txn_seq = 0
recent_by_account = {}

for day in range(30):
    day_start = base_day + timedelta(days=day)
    for _ in range(700):
        account = random.choice(accounts)
        ts = day_start + timedelta(seconds=random.randint(0, 86399))
        pattern = None
        roll = random.random()
        if roll < 0.015:
            pattern = "large_foreign_cnp"
        elif roll < 0.03:
            pattern = "amount_outlier"

        txn_seq += 1
        txn = make_transaction(f"T{txn_seq:07d}", account, ts, pattern)

        bucket = recent_by_account.setdefault(account["account_id"], [])
        bucket = [t for t in bucket if (ts - t).total_seconds() <= 600]
        velocity_count = len(bucket) + 1
        bucket.append(ts)
        recent_by_account[account["account_id"]] = bucket[-10:]

        txn["is_fraud"] = label_is_fraud(txn, account, velocity_count, random.random())
        labeled.append(txn)

write_json_lines(f"{RAW_ROOT}/transactions_labeled/labeled_batch_000.json", labeled)
print(f"Labeled historical transactions: {len(labeled)}, fraud rate: {sum(t['is_fraud'] for t in labeled) / len(labeled):.3f}")

# COMMAND ----------
# unlabeled streaming batches (the live feed the pipeline ingests + scores)
stream_start = base_day + timedelta(days=30)
num_stream_batches, txns_per_batch = 8, 250

for batch in range(num_stream_batches):
    batch_ts = stream_start + timedelta(minutes=batch * 20)
    batch_txns = []
    for _ in range(txns_per_batch):
        account = random.choice(accounts)
        ts = batch_ts + timedelta(seconds=random.randint(0, 1199))
        pattern = None
        roll = random.random()
        if roll < 0.02:
            pattern = "large_foreign_cnp"
        elif roll < 0.035:
            pattern = "amount_outlier"

        txn_seq += 1
        txn = make_transaction(f"T{txn_seq:07d}", account, ts, pattern)
        txn["is_fraud"] = None  # unknown -- this is what the pipeline will predict
        batch_txns.append(txn)

    write_json_lines(f"{RAW_ROOT}/transactions_stream/stream_batch_{batch:03d}.json", batch_txns)

print(f"Sample data generated under {RAW_ROOT}")
