#!/bin/bash
# Create Kubernetes secret from .env file

# Source the .env file to get the credentials
source ../../.env

# Create the secret
kubectl create secret generic s3-credentials \
  --from-literal=access-key-id="$AWS_ACCESS_KEY_ID" \
  --from-literal=secret-access-key="$AWS_SECRET_ACCESS_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secret 's3-credentials' created successfully"
