```
docker build . -t rltools/ui-server
```
```
docker run -it --rm -p 8000:8000 rltools/ui-server
```

```
docker run -it --rm -p 8000:8000 rltools/ui-server 0.0.0.0 8000
```
