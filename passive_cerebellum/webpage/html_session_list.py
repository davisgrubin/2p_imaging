#!/usr/bin/env python3

def get_session_list_html(session_config_list, list_target_sess):
    # classify sessions.
    sess_name = {name: [] for name in list_target_sess}
    for k, v in session_config_list['list_session_name'].items():
        sess_name.setdefault(v, []).append(k)
    # calculate total numbers.
    sess_name = {k+f' ({len(sess_name[k])})': v for k,v in sess_name.items()}
    # create table.
    table_headers = "".join(f"<th>{col}</th>" for col in sess_name.keys())
    max_rows = max(len(items) for items in sess_name.values())
    table_rows = ""
    for row in range(max_rows):
        table_rows += "<tr>"
        for col, items in sess_name.items():
            cell = items[row] if row < len(items) else ""
            table_rows += f"<td>{cell}</td>"
        table_rows += "</tr>"
    # write into html.
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