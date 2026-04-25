s3-up:
	kubectl config use-context docker-desktop
	./create-s3-secret.sh
	kubectl apply -f s3proxy-deployment.yaml

	# Wait for deployment to be ready
	kubectl wait --for=condition=Ready --timeout=300s pods/s3proxy-pod
	kubectl get pod
	aws s3 ls 
	kubectl logs -l app=s3proxy

s3-logs:
	kubectl logs -f deployment/s3proxy-deployment -f

s3-down:
	kubectl delete -f s3proxy-deployment.yaml
	kubectl delete secret s3-credentials

ray-up:
	kubectl config use-context docker-desktop  
	kubectl apply -f https://github.com/ray-project/kuberay/releases/latest/download   │ │   /kuberay-operator.yaml
	kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?re   │
│   f=v1.0.0&timeout=90s"

s3-nrp-ls:
	aws s3 ls --profile=braingeneers s3://braingeneers/personal/rcurrie/