#!/usr/bin/env python3
import http.server
import socketserver
import os

os.chdir('D:/assortments/GraphRAG/Fin/visualization')

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    with socketserver.TCPServer(('127.0.0.1', 8080), CORSRequestHandler) as httpd:
        print('Serving on http://127.0.0.1:8080')
        print('Open: http://127.0.0.1:8080/graph_visualization.html')
        httpd.serve_forever()
