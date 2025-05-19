#!/usr/bin/env python3

def get_session_list_html(session_config):
    sess_name = {
        'single':[]}
    for key, value in session_config['list_session_name'].items():
        sess_name.setdefault(value, []).append(key)
    sess_name = {
        'single ({})'.format(len(sess_name['single'])): sess_name['single'],
        }
    table_headers = "".join(f"<th>{col}</th>" for col in sess_name.keys())
    max_rows = max(len(items) for items in sess_name.values())
    table_rows = ""
    for row in range(max_rows):
        table_rows += "<tr>"
        for col, items in sess_name.items():
            cell = items[row] if row < len(items) else ""
            table_rows += f"<td>{cell}</td>"
        table_rows += "</tr>"
    session_list_html = f"""
    <table class="info-table">
        <thead>
            <tr>{table_headers}</tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    """
    return session_list_html