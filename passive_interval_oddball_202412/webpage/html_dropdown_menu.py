#!/usr/bin/env python3

def get_dropdown_menu_html():
    labels = ['field_of_view', 'random', 'short_long', 'fix_jitter_odd']
    n_pages=4
    dropdown_options = ""
    for i in range(n_pages):
        checked = "checked" if i == 0 else ""
        dropdown_options += f"""
        <label>
            <input 
                type="radio"
                name="pageRadio"
                value="page{i+1}"
                onchange="radioChanged(this)"
                {checked}>
                {labels[i]}
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