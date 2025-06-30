from waitress import serve
from a2wsgi import ASGIMiddleware
from scraper_service.app import app

wsgi_app = ASGIMiddleware(app)

if __name__ == "__main__":
    print("Attempting to start server with Waitress on http://127.0.0.1:8888")
    serve(wsgi_app, host="127.0.0.1", port=8888)