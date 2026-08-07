from flask import Flask, request, Response
from flask_cors import CORS
from superlm import WorkSpace
from ast import literal_eval
from functools import lru_cache

app = Flask(__name__)
CORS(app)

@lru_cache(maxsize=3)
def get_workspace(name: str) -> WorkSpace:
    return WorkSpace(name)

def sse(x):
    with app.app_context():
        return f'data: {x}\n\n'

@app.route('/api/<name>', methods=['GET', 'POST'])
def api(name: str) -> Response:
    workspace = get_workspace(name)
    if request.method == 'GET':
        args = dict(request.args)
    else:
        args = dict(request.get_json())
    prompt = args.pop('prompt', '')
    for k, v in args.copy().items():
        args[k] = literal_eval(v)
    return Response(
        workspace.api(prompt, sse, **args),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache'},
    )

if __name__ == '__main__':
    app.run(debug=True)