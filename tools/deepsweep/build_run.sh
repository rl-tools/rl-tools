docker build -t rltools/sweep .
docker push rltools/sweep
docker run -it --rm -p 13338:13338 rltools/sweep