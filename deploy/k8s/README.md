# Kubernetes 매니페스트 (k3s 대상, 정적 검증 단계)

## 솔직한 범위 표기

이 디렉토리의 매니페스트는 **단일 GPU 노드 k3s를 대상으로 작성**했고,
현재까지 검증한 것은 **YAML 스키마와 리소스 간 참조 무결성뿐**이다.
아직 클러스터에 적용해 보지 않았다 — 아래 "적용" 절차는 실행 기록이 아니라
실행 방법이다.

이력서나 발표에서는 "K8s 운영 경험"이 아니라 **"배포 매니페스트 작성 및
정적 검증"** 으로 표기해야 정확하다. 클러스터에 실제로 올린 뒤에야
표현을 올릴 수 있다.

단일 GPU 한 장에 Kubernetes는 사실 과한 도구다. 그럼에도 매니페스트를 둔 이유는
docker-compose에서는 드러나지 않는 제약을 명시적으로 다루기 위해서다:

| 제약 | compose | k8s |
|---|---|---|
| GPU를 한 파드만 점유 | 암묵적 (컨테이너 하나만 띄움) | `nvidia.com/gpu: 1` 리소스로 스케줄러가 강제 |
| 모델 로딩 180초 동안 트래픽 차단 | healthcheck만 있음 | `readinessProbe`가 Service 엔드포인트에서 제외 |
| UI만 수평 확장 | 불가 | `HorizontalPodAutoscaler`를 UI Deployment에만 적용 |
| 가중치 공유 | 호스트 바인드 마운트 | `PersistentVolumeClaim` (ReadOnlyMany) |

## 구성 파일

| 파일 | 내용 |
|---|---|
| `namespace.yaml` | `fusiondti` 네임스페이스 |
| `configmap.yaml` | 백엔드 URL, MLflow URI 등 비민감 설정 |
| `secret.yaml.example` | Postgres 비밀번호 — 실제 `secret.yaml`은 커밋하지 않는다 |
| `pvc.yaml` | 모델 가중치(`results/`) + HuggingFace 캐시용 볼륨 |
| `api-deployment.yaml` | GPU 파드 1개 (replicas 고정), readiness/liveness probe |
| `ui-deployment.yaml` | CPU 파드, HPA 대상 |
| `hpa.yaml` | UI만 CPU 60% 기준 2~5 replica |
| `ingress.yaml` | Traefik(k3s 기본) 경로 라우팅 |

## 적용

```bash
# k3s 설치 (GPU 지원)
curl -sfL https://get.k3s.io | sh -
sudo k3s kubectl get nodes

# NVIDIA device plugin (GPU 리소스를 스케줄러에 노출)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml

# 이미지를 k3s 내장 containerd로 가져오기 (레지스트리 없이)
docker save fusiondti-api:latest | sudo k3s ctr images import -
docker save fusiondti-ui:latest  | sudo k3s ctr images import -

# 배포
kubectl apply -f k8s/namespace.yaml
cp deploy/k8s/secret.yaml.example deploy/k8s/secret.yaml && $EDITOR deploy/k8s/secret.yaml
kubectl apply -f deploy/k8s/

# 확인
kubectl -n fusiondti get pods,svc,hpa
kubectl -n fusiondti logs -f deploy/fusiondti-api
```

## 알려진 한계

- **GPU 파드는 스케일 아웃되지 않는다.** replicas를 2로 올리면 두 번째 파드가
  `Pending`에 머문다 (할당 가능한 `nvidia.com/gpu`가 1개뿐). 의도된 동작이며,
  `api-deployment.yaml`의 replicas는 1로 고정해 두었다.
- **PVC는 k3s 기본 `local-path` 프로비저너를 쓴다.** 단일 노드라 성립하는
  구성이고, 다중 노드에서는 ReadWriteMany를 지원하는 스토리지(NFS 등)가 필요하다.
- **Ingress는 TLS 없이 HTTP로만 노출한다.** 로컬 검증 목적이다.
