wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uk0Paw_wLg4qKZFHAszREkSde6tarT6c' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uk0Paw_wLg4qKZFHAszREkSde6tarT6c" -O fcn8.zip && rm -rf /tmp/cookies.txt
unzip fcn8.zip

python3 problem_2/test.py --checkpoint problem_2/checkpoints/fcn8_v4/fcn8_best.pth --test-dir $1 --output-dir $2