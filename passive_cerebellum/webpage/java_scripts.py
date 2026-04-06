#!/usr/bin/env python3

def get_java_scripts(n_pages):
    all_pages_js = "[" + ",".join(f'"page{i+1}"' for i in range(n_pages)) + "]"
    java_scripts = f"""
    <script type="text/javascript">
        function toggleDropdown() {{
            document.getElementById("myDropdown").classList.toggle("show");
            }}
            window.onclick = function(event) {{
            if (!event.target.matches('.dropbtn')) {{
                var dropdown = document.getElementById("myDropdown");
                if (dropdown.classList.contains('show')) {{
                    dropdown.classList.remove('show');
                }}
            }}
        }}
        function radioChanged(radio) {{
            var allOptions = {all_pages_js};
            for (var i = 0; i < allOptions.length; i++) {{
                var pageDiv = document.getElementById(allOptions[i]);
                if (allOptions[i] === radio.value) {{
                    pageDiv.style.display = 'block';
                }} else {{
                    pageDiv.style.display = 'none';
                }}
            }}
        }}
    </script>
    """
    return java_scripts