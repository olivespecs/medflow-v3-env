import os
import zipfile
import fnmatch
from pathlib import Path

def get_ignore_patterns(gitignore_path):
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    return patterns

def is_ignored(path, root, patterns):
    rel_path = os.path.relpath(path, root)
    if rel_path == '.':
        return False
        
    rel_path_standard = rel_path.replace(os.sep, '/')
    
    for pattern in patterns:
        p = pattern.rstrip('/')
        if pattern.endswith('/'):
            if rel_path_standard.startswith(p + '/') or rel_path_standard == p:
                return True
        
        if fnmatch.fnmatch(rel_path_standard, p) or fnmatch.fnmatch(os.path.basename(rel_path_standard), p):
            return True
            
        parts = rel_path_standard.split('/')
        for i in range(len(parts)):
            sub_path = '/'.join(parts[:i+1])
            if fnmatch.fnmatch(sub_path, p):
                return True
                
        if '**' in pattern:
            if fnmatch.fnmatch(rel_path_standard, pattern):
                return True
    return False

def compress_project(root_dir, output_zip, ignore_patterns):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            if is_ignored(root, root_dir, ignore_patterns):
                dirs[:] = []
                continue
                
            dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), root_dir, ignore_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                if not is_ignored(file_path, root_dir, ignore_patterns):
                    rel_path = os.path.relpath(file_path, root_dir)
                    print(f"Adding: {rel_path}")
                    zipf.write(file_path, rel_path)

if __name__ == "__main__":
    project_root = r"c:\Users\yasha\Desktop\hackathon"
    output_filename = os.path.join(project_root, "hackathon_submission.zip")
    
    additional_ignores = ["compress_project.py", "hackathon_submission.zip", ".git/"]
    ignore_patterns = get_ignore_patterns(os.path.join(project_root, ".gitignore"))
    ignore_patterns.extend(additional_ignores)
    
    if os.path.exists(output_filename):
        os.remove(output_filename)
        
    compress_project(project_root, output_filename, ignore_patterns)
    print(f"Done! Created {output_filename}")
