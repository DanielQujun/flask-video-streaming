docker build . \
    --build-arg "HTTPS_PROXY=http://192.168.6.221:7890" \
    --build-arg "HTTP_PROXY=http://192.168.6.221:7890" \
    --build-arg "https_proxy=http://192.168.6.221:7890" \
    --build-arg "http_proxy=http://192.168.6.221:7890" \
    --build-arg "NO_PROXY=localhost,127.0.0.1,.example.com" \
    --build-arg "no_proxy=localhost,127.0.0.1,.example.com" \
    -t miniforge3_camera_x86:latest -f Dockerfile_base

