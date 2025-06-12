# SPDX-License-Identifier: Apache-2.0
import os

import yaml

config_file = os.environ["CONFIG_FILE"]
test_name = os.environ["TEST_NAME"]

with open(config_file) as f:
    config = yaml.safe_load(f)

test_vars = config[test_name]

with open(".env.generated", "w") as f:
    for k, v in test_vars.items():
        f.write(f"{k}={v}\n")