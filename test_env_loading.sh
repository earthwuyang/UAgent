#!/bin/bash

# Load the .env file
set -a
source .env
set +a

# Show the value of the proxy configuration
echo "SANDBOX_RUNTIME_STARTUP_ENV_VARS value:"
echo "$SANDBOX_RUNTIME_STARTUP_ENV_VARS"
echo ""

# Test Python parsing
python3 -c "
import os
import ast

env_vars = os.environ.get('SANDBOX_RUNTIME_STARTUP_ENV_VARS', '{}')
print('Raw value:', repr(env_vars))
print('')

try:
    parsed = ast.literal_eval(env_vars)
    print('Parsed successfully:')
    for key, value in parsed.items():
        print(f'  {key} = {value}')
except Exception as e:
    print(f'Failed to parse: {e}')
"