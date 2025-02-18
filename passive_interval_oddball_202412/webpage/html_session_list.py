#!/usr/bin/env python3

def get_session_list_html(session_config):
    sess_name = {
        'random':[],
        'short_long':[],
        'fix_jitter_odd':[]}
    for key, value in session_config['list_session_name'].items():
        sess_name.setdefault(value, []).append(key)
    sess_name = {
        'random ({})'.format(len(sess_name['random'])): sess_name['random'],
        'short_long ({})'.format(len(sess_name['short_long'])): sess_name['short_long'],
        'fix_jitter_odd ({})'.format(len(sess_name['fix_jitter_odd'])): sess_name['fix_jitter_odd'],
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