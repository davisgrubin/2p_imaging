#!/usr/bin/env python3

def get_css_styles():
    css_styles = """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            position: relative;
            overflow-x: auto;
          }
          h1, h2 {
            color: #333;
          }
          .page-container {
            margin-top: 20px;
          }
          .figure-block {
            display: flex;
            flex-direction: row;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 10px;
            margin: 10px 0;
            position: relative;
            min-width: 4320px;
          }
          .figure-block:hover {
            box-shadow: 0 0 10px rgba(0,0,0,0.15);
          }
          .left-panel {
            width: 360px;
            margin-right: 20px;
            border-right: 1px solid #ccc;
            padding-right: 10px;
          }
          .description {
            font-family: "Times New Roman", serif;
            font-size: 11px;
            line-height: 11px;
            color: #444;
            text-align: justify;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow: hidden;
            padding: 5px;
          }
          .right-panel {
            flex: 1 1 auto;
          }
          .figure-title {
            margin-bottom: 5px;
          }
          .download-title {
            background-color: black;
            color: white;
            text-decoration: none;
            padding: 6px 10px;
            border-radius: 4px;
            display: inline-block;
          }
          .download-title:hover {
            opacity: 0.8;
          }
          .svg-container {
            width: 100%;
            height: auto;
          }
          .svg-container svg {
            height: 100%;
            width: auto;
          }
          /* Dropdown Styles */
          .dropdown {
            position: relative;
            display: inline-block;
            margin-bottom: 10px;
          }
          .dropbtn {
            background-color: orange;
            color: white;
            padding: 8px 12px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
          }
          .dropdown-content {
            display: none;
            position: absolute;
            background-color: #f9f9f9;
            min-width: 160px;
            max-height: 200px;
            overflow-y: auto;
            box-shadow: 0px 8px 16px rgba(0,0,0,0.2);
            z-index: 1;
            border-radius: 4px;
            padding: 10px;
          }
          .dropdown-content label {
            display: block;
            padding: 4px;
            color: #333;
            cursor: pointer;
          }
          .dropdown-content label:hover {
            background-color: #ddd;
          }
          .show {
            display: block;
          }
          /* Table Styles */
          .info-table {
            width: 2500 px;
            border-collapse: collapse;
            margin: 20px 0;
          }
          .info-table th, .info-table td {
            border-right: 1px solid #ccc;
            vertical-align: top;
            padding: 8px;
            text-align: left;
          }
          .info-table th:last-child, .info-table td:last-child {
            border-right: none;
          }
          .info-table th {
            font-weight: bold;
          }
        </style>
        """
    return css_styles
        