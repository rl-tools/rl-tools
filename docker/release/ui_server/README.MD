In the main RLtools directory
```
docker build . -f docker/release/ui_server/Dockerfile --platform linux/amd64 -t rltools/ui-server
```
```
docker run -it --rm -p 13337:13337 rltools/ui-server
```
```
docker tag rltools/ui-server rltools/ui-server:latest
docker push rltools/ui-server:latest
docker tag rltools/ui-server rltools/ui-server:2.0.0
docker push rltools/ui-server:2.0.0
```
```

```
docker run -it --rm -p 13337:13337 rltools/ui-server 0.0.0.0 13337 multirotor default
```
```
docker push rltools/ui-server
```
