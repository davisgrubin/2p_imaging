#!/usr/bin/env python3

def get_dropdown_menu_html(list_page_name):
    dropdown_options = ""
    for i in range(len(list_page_name)):
        checked = "checked" if i == 0 else ""
        dropdown_options += f"""
        <label>
            <input 
                type="radio"
                name="pageRadio"
                value="page{i+1}"
                onchange="radioChanged(this)"
                {checked}>
                {list_page_name[i]}
                </label>
        """
    dropdown_html = f"""
    <div class="dropdown">
        <button onclick="toggleDropdown()" class="dropbtn">Select Page</button>
        <div id="myDropdown" class="dropdown-content">
            {dropdown_options}
        </div>
    </div>
    <p style="color: #666; font-size: 0.9em;">
        Select page to continue. Click figure titles to download PDF files.
    </p>
    """
    return dropdown_html