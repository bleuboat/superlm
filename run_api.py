from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from superlm import WorkSpace
from ast import literal_eval

app = Flask(__name__)
CORS(app)
workspaces = {}

def sse(x):
    with app.app_context():
        return f'data: {x}\n\n'

@app.route('/api/<name>', methods=['GET', 'POST'])
def api(name: str) -> Response:
    workspace = workspaces.get(name, None)
    if workspace is None:
        workspace = WorkSpace(name)
        workspaces[name] = workspace
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