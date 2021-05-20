FROM python:3.8.3

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update
#ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#RUN apt-get install -y zbar-tools libzbar-dev libzbar0


RUN pip install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# NOTES:
# According to  gunicorn documentation
# https://docs.gunicorn.org/en/stable/settings.html#workers
# A positive integer generally in the 2-4 x $(NUM_CORES) range.
# we are running every cloud run with 2 cpu cores
# as result setting up 8 workers
# according to the documentation:
# https://docs.gunicorn.org/en/stable/settings.html#threads
# A positive integer generally in the 2-4 x $(NUM_CORES) range.
# as result setting up 8 threads
# setting --max-requests as per documentation it helps prevent memory leaks
# https://docs.gunicorn.org/en/stable/settings.html#workers
CMD exec gunicorn --bind :$PORT --workers 8 --threads 8 --timeout 0 --max-requests-jitter 10 app:app