import os
import json

db_filename = "out/db.json"


def save_to_db(key, value):
    # Hack. switch to redis/sqlite at some point.
    if not os.path.exists(db_filename):
        db = {key: value}
        with open(db_filename, "w") as f:
            json.dump(db, f, indent=4)
        return

    with open(db_filename, "r") as f:
        db = json.load(f)
    db[key] = value
    with open(db_filename, "w") as f:
        json.dump(db, f, indent=2)
