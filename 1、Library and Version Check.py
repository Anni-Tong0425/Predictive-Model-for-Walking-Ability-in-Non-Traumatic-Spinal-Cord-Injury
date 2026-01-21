# -*- coding: utf-8 -*-
"""
Bernoulli Naive Bayes Project - Library Version Checker
检查项目中使用的所有库的版本号
"""

import sys
import platform
import subprocess
import json


def check_versions():
    """检查项目中使用的所有库的版本"""

    print("=" * 80)
    print("BERNOULLI NAIVE BAYES PROJECT - VERSION INFORMATION")
    print("=" * 80)

    # Python信息
    print("\n1. PYTHON INFORMATION:")
    print("-" * 40)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Python Full Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Implementation: {platform.python_implementation()}")

    # 项目使用的库列表
    project_libraries = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'shap',
        'joblib',
        'statsmodels'  # 可能在计算中使用
    ]

    print(f"\n2. PROJECT LIBRARIES ({len(project_libraries)} total):")
    print("-" * 40)

    # 使用pip show获取详细信息
    library_info = []

    for lib in project_libraries:
        try:
            # 尝试使用pip show获取库信息
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', lib],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                info_lines = result.stdout.strip().split('\n')
                version = None
                location = None

                for line in info_lines:
                    if line.startswith('Version:'):
                        version = line.split(': ')[1]
                    elif line.startswith('Location:'):
                        location = line.split(': ')[1]

                library_info.append({
                    'library': lib,
                    'version': version,
                    'location': location,
                    'status': 'Installed'
                })

                print(f"✓ {lib:<20} {version if version else 'N/A':<15}")
            else:
                library_info.append({
                    'library': lib,
                    'version': 'Not Found',
                    'location': 'N/A',
                    'status': 'Not Installed'
                })
                print(f"✗ {lib:<20} {'Not Installed':<15}")

        except Exception as e:
            library_info.append({
                'library': lib,
                'version': f'Error: {str(e)}',
                'location': 'N/A',
                'status': 'Error'
            })
            print(f"✗ {lib:<20} {'Error':<15}")

    # 检查其他可能的内置库
    print(f"\n3. BUILT-IN LIBRARIES:")
    print("-" * 40)
    builtin_libs = ['pickle', 'time', 'os', 'warnings', 'json', 'sys']
    for lib in builtin_libs:
        print(f"✓ {lib:<20} {'Built-in':<15}")

    # 尝试导入库并检查实际可用性
    print(f"\n4. LIBRARY IMPORT TEST:")
    print("-" * 40)

    import_tests = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
        ('shap', 'shap'),
        ('joblib', 'joblib')
    ]

    for lib_name, lib_alias in import_tests:
        try:
            __import__(lib_name)
            print(f"✓ {lib_name:<20} Import successful")
        except ImportError as e:
            print(f"✗ {lib_name:<20} Import failed: {str(e)}")
        except Exception as e:
            print(f"✗ {lib_name:<20} Error: {str(e)}")

    # 生成详细报告
    print(f"\n5. DETAILED VERSION REPORT:")
    print("-" * 40)

    for info in library_info:
        if info['status'] == 'Installed':
            print(f"\n{info['library']}:")
            print(f"  Version: {info['version']}")
            print(f"  Location: {info['location'][:50]}..." if len(
                info['location']) > 50 else f"  Location: {info['location']}")

    # 保存到JSON文件
    report_data = {
        'python_version': sys.version,
        'python_platform': platform.platform(),
        'libraries': library_info,
        'timestamp': platform.python_implementation()
    }

    try:
        with open('project_versions.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\n✓ Version report saved to: project_versions.json")
    except Exception as e:
        print(f"\n✗ Failed to save version report: {e}")

    # 生成requirements.txt
    print(f"\n6. GENERATING REQUIREMENTS.TXT:")
    print("-" * 40)

    try:
        with open('requirements_project.txt', 'w') as f:
            f.write(f"# Requirements for Bernoulli Naive Bayes Project\n")
            f.write(f"# Generated automatically\n")
            f.write(f"# Python {sys.version.split()[0]}\n\n")

            for info in library_info:
                if info['status'] == 'Installed' and info['version'] != 'Not Found':
                    # 特殊处理scikit-learn的包名
                    if info['library'] == 'scikit-learn':
                        f.write(f"scikit-learn=={info['version']}\n")
                    else:
                        f.write(f"{info['library']}=={info['version']}\n")

            # 添加一些可能有用的库
            f.write("\n# Optional but useful libraries\n")
            f.write("# shap  # For model interpretability\n")
            f.write("# seaborn  # For advanced visualizations\n")

        print(f"✓ Requirements file generated: requirements_project.txt")

    except Exception as e:
        print(f"✗ Failed to generate requirements file: {e}")

    print("\n" + "=" * 80)
    print("VERSION CHECK COMPLETE!")
    print("=" * 80)


def check_specific_versions():
    """检查特定库的详细版本信息"""

    print("\n" + "=" * 80)
    print("DETAILED VERSION CHECK FOR KEY LIBRARIES")
    print("=" * 80)

    libraries_to_check = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn'),
        ('shap', 'shap')
    ]

    for lib_name, lib_alias in libraries_to_check:
        print(f"\n{lib_name.upper()}:")
        print("-" * 40)

        try:
            # 导入库
            module = __import__(lib_name)

            # 获取版本
            if hasattr(module, '__version__'):
                version = module.__version__
            elif hasattr(module, 'version'):
                version = module.version
            else:
                version = 'Unknown'

            print(f"Version: {version}")

            # 获取其他信息
            if lib_name == 'numpy':
                import numpy as np
                print(f"Configuration: {np.show_config()}")

            elif lib_name == 'pandas':
                import pandas as pd
                print(f"Pandas options:")
                print(f"  display.max_columns: {pd.get_option('display.max_columns')}")
                print(f"  display.max_rows: {pd.get_option('display.max_rows')}")

            elif lib_name == 'matplotlib':
                import matplotlib as mpl
                print(f"Backend: {mpl.get_backend()}")
                print(f"Matplotlib location: {mpl.__file__}")

            elif lib_name == 'scikit-learn':
                import sklearn
                print(f"Scikit-learn location: {sklearn.__file__}")

        except ImportError:
            print(f"ERROR: {lib_name} is not installed")
        except Exception as e:
            print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    check_versions()
    check_specific_versions()

    # 提供安装建议
    print("\n" + "=" * 80)
    print("INSTALLATION SUGGESTIONS:")
    print("=" * 80)
    print("\nIf any libraries are missing, you can install them using:")
    print("\npip install numpy pandas matplotlib seaborn scipy scikit-learn shap joblib")
    print("\nOr create a virtual environment and install from requirements_project.txt:")
    print("\npip install -r requirements_project.txt")