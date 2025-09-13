About This Package
This package functions as executor to create, insert, vectorized and search on tidb database. Shortly 
could said this is a database connector to Tidb cloud.

🚀 How to Run It

📥 Download

```bash
docker pull ghcr.io/hendram/vectorembedgen
```

▶️ Start

```bash
docker run -it -d --network=host ghcr.io/hendram/vectorembedgen bash
```

🔍 Check Running Container

```bash
docker ps
```

```bash
CONTAINER ID   IMAGE                               NAME                STATUS
123abc456def   ghcr.io/hendram/vectorembedgen      confident_banzai    Up 5 minutes
```

📦 Enter Container

```bash
docker exec -it confident_banzai /bin/bash
```

🏃 Run the Service

```bash
cd /home
source .venv/bin/activate
uvicorn vectorembedgen:app --host 0.0.0.0 --port 8000
```
