import sys
import json
import os
import logging
from uuid import uuid4
from flask import Flask
from flask import request
from numpy import argmax

from load_models import clf_monster, textVectorizer
from utils import Utils
from carmilla_copilot import CarmillaCopilot

"""
Define simple logger
"""
logFormatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] \
    [%(filename)s:%(lineno)s - %(funcName)20s() ] \
    %(message)s')

log = logging.getLogger(__name__)

stdLogger = logging.StreamHandler(sys.stdout)
stdLogger.setFormatter(logFormatter)
log.addHandler(stdLogger)
log.setLevel(logging.DEBUG)


app = Flask(__name__)
carmilla = CarmillaCopilot(log)
"""Setup initial requirements"""

@app.route('/api/carmilla', methods=['GET'])
def carmilla_copilot():
    """Continue in the style of Carmilla"""
    log.info("STARTED")
    id_request = uuid4().hex
    # log headers and IP
    try:
        remote_addr = str(request.remote_addr)
        headers = str(request.headers)
        host = request.headers.get("Host")
        user_agent = request.headers.get("User-Agent")
        cookie = request.headers.get("Cookie")
        log.info(f"IP:{remote_addr} headers:{headers}")
    except Exception as e:
        log.error(f"logging of headers failed with: {e}")

    try:
        text = str(request.args.get("x"))
        text = text.replace("%20", " ")
        log.debug(f"text: {text}")
        output_text = carmilla(text)

        output = {
            "id_request": id_request,
            "remote_addr": remote_addr,
            "user_agent": user_agent,
            "cookie": cookie,
            "input_text": text,
            "carmilla_copilot": output_text,
        }
            
        response = app.response_class(
            response=json.dumps(output),
            status=200,
            mimetype='application/json'
        )

        return response
    except Exception as e:
        log.error(e)
        return Utils.error_response(id_request, log)


@app.route('/api/vampire', methods=['GET'])  # type: ignore
def vampire_call():
    """Vampiric classification"""
    log.info("STARTED")
    id_request = uuid4().hex
    # log headers and IP
    try:
        remote_addr = str(request.remote_addr)
        headers = str(request.headers)
        host = request.headers.get("Host")
        user_agent = request.headers.get("User-Agent")
        cookie = request.headers.get("Cookie")

        log.info(f"IP:{remote_addr} headers:{headers}")
    except Exception as e:
        log.error(f"logging of headers failed with: {e}")

    try:
        text = str(request.args.get("x"))

        text = text.replace("%20", " ")
        log.debug(f"text: {text}")

        text_vector = [textVectorizer.to_count_vec(text)]
        bow = textVectorizer(text)

        monster_type_prob = clf_monster.predict_proba(text_vector)[0]
        monster_dct = {1: "Vampire", 0: "Frankstein's"}
        prob_max = max(monster_type_prob)
        monster = monster_dct.get(
            argmax(monster_type_prob), "not sure")

        output = {
            "id_request": id_request,
            "remote_addr": remote_addr,
            "user_agent": user_agent,
            "cookie": cookie,
            "input_text": text,
            "bag_of_words": str(bow),
            "monster_type_prediction": monster,
            "prediction_probability": prob_max
        }
            
        response = app.response_class(
            response=json.dumps(output),
            status=200,
            mimetype='application/json'
        )

        return response
    except Exception as e:
        log.error(e)
        return Utils.error_response(id_request, log)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


def log_headers_and_ip(_request):
    log.debug("started log for headers/ip")
    try:
        log.info(f"IP:{_request.remote_addr} headers:{_request.headers}")
    except Exception:
        log.info("unable to log headers and IP")


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.

    app.run(host='127.0.0.1', port=8080, debug=True)
