```
docker build . --platform linux/amd64 -t rltools/ui-server
```
```
docker run -it --rm -p 8000:8000 rltools/ui-server
```

```
docker run -it --rm -p 8000:8000 rltools/ui-server 0.0.0.0 8000 multirotor default
```
```
docker push rltools/ui-server
```
