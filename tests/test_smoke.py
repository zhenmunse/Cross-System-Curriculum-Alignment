# -*- coding: utf-8 -*-
"""
冒烟测试：确保基本流程可以跑通。
"""
import unittest
import os
import subprocess
import sys

class TestSmoke(unittest.TestCase):

    def setUp(self):
        """测试前的准备工作。"""
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.results_dir = os.path.join(self.project_root, 'results')
        # 在运行前清理旧的结果文件
        if os.path.exists(self.results_dir):
            for f in os.listdir(self.results_dir):
                os.remove(os.path.join(self.results_dir, f))

    def test_run_main_script(self):
        """
        测试：执行主脚本，并断言结果文件已生成且非空。
        """
        # 构建命令行
        main_script_path = os.path.join(self.project_root, 'main.py')
        cmd = [
            sys.executable,  # 使用当前 Python 解释器
            main_script_path,
            '--los', os.path.join(self.project_root, 'data/learning_outcomes.csv'),
            '--modules', os.path.join(self.project_root, 'data/modules.csv'),
            '--source_system', 'A-level',
            '--target_system', 'University',
            '--out_dir', self.results_dir
        ]

        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        # 打印输出以便调试
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)

        # 断言命令成功执行
        self.assertEqual(result.returncode, 0, "主脚本执行失败")

        # 定义预期的输出文件
        expected_files = [
            'mappings_a-level_to_university.csv',
            'metrics_a-level_to_university.csv',
            'coverage_heatmap_a-level_to_university.png',
            'gap_distribution_university.png',
            'pair_type_pie.png',
            'run.log'
        ]

        # 断言所有预期文件都已生成
        for filename in expected_files:
            filepath = os.path.join(self.results_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"结果文件未生成: {filename}")
            # 断言文件非空
            self.assertGreater(os.path.getsize(filepath), 0, f"结果文件为空: {filename}")

if __name__ == '__main__':
    unittest.main()
