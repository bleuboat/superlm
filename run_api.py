from flask import Flask, request, Response
from flask_cors import CORS
from superlm import WorkSpace

app = Flask(__name__)
CORS(app)
workspaces = {}

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
    return Response(
        workspace.api(prompt, **args),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache'},
    )

if __name__ == '__main__':
    app.run()