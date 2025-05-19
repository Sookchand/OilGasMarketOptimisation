"""
Script to update all instances of st.experimental_rerun() to st.rerun() in enhanced_app.py
"""

def update_file(file_path):
    try:
        # Read the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Count occurrences before replacement
        count_before = content.count('st.experimental_rerun()')
        print(f"Found {count_before} instances of st.experimental_rerun()")

        # Replace all instances of st.experimental_rerun() with st.rerun()
        updated_content = content.replace('st.experimental_rerun()', 'st.rerun()')

        # Count occurrences after replacement
        count_after = updated_content.count('st.rerun()')
        print(f"Replaced with {count_after} instances of st.rerun()")

        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

        print(f"Updated {file_path}")
        return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

if __name__ == "__main__":
    update_file('enhanced_app.py')
