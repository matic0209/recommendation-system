#!/usr/bin/env python3
"""
验证推荐系统埋点和数据收集机制

检查项：
1. API曝光日志是否正常记录
2. Matomo数据是否能正确读取
3. 评估pipeline是否能正常计算CTR/CVR
4. 数据目录结构是否完整
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR


class TrackingVerifier:
    """埋点验证器"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.exposure_log = DATA_DIR / "evaluation" / "exposure_log.jsonl"
        self.results: List[Dict] = []

    def check(self, name: str, passed: bool, message: str, details: Optional[str] = None):
        """记录检查结果"""
        status = "✓ PASS" if passed else "✗ FAIL"
        result = {
            "name": name,
            "passed": passed,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)

        print(f"{status} | {name}")
        print(f"     {message}")
        if details:
            print(f"     详情: {details}")
        print()

    def verify_api_health(self) -> bool:
        """检查API服务健康状态"""
        print("=" * 60)
        print("1. API服务健康检查")
        print("=" * 60)

        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            if resp.status_code == 200:
                health_data = resp.json()
                self.check(
                    "API健康检查",
                    True,
                    "API服务正常运行",
                    f"状态: {health_data.get('status')}, 模型: {health_data.get('models_loaded')}"
                )
                return True
            else:
                self.check(
                    "API健康检查",
                    False,
                    f"API返回异常状态码: {resp.status_code}",
                    resp.text[:200]
                )
                return False
        except Exception as e:
            self.check(
                "API健康检查",
                False,
                f"无法连接到API: {e}",
                f"请确保API服务运行在 {self.api_url}"
            )
            return False

    def verify_exposure_logging(self) -> bool:
        """验证曝光日志记录功能"""
        print("=" * 60)
        print("2. 曝光日志记录验证")
        print("=" * 60)

        # 检查目录是否存在
        exposure_dir = self.exposure_log.parent
        if not exposure_dir.exists():
            self.check(
                "曝光日志目录",
                False,
                f"曝光日志目录不存在: {exposure_dir}",
                "运行一次pipeline或手动创建目录"
            )
            return False

        # 记录调用前的日志文件状态
        log_existed = self.exposure_log.exists()
        if log_existed:
            initial_size = self.exposure_log.stat().st_size
            with open(self.exposure_log, 'r') as f:
                initial_lines = len(f.readlines())
        else:
            initial_size = 0
            initial_lines = 0

        self.check(
            "曝光日志文件",
            True,
            f"当前日志文件: {'存在' if log_existed else '不存在'}",
            f"大小: {initial_size} bytes, 行数: {initial_lines}"
        )

        # 调用API生成曝光日志
        print("调用API生成测试曝光...")
        test_requests = []

        # 测试1: /similar接口
        try:
            resp = requests.get(f"{self.api_url}/similar/1?limit=5", timeout=5)
            if resp.status_code == 200:
                test_requests.append(("similar", True, resp.json()))
            else:
                test_requests.append(("similar", False, f"状态码: {resp.status_code}"))
        except Exception as e:
            test_requests.append(("similar", False, str(e)))

        # 测试2: /recommend/detail接口
        try:
            resp = requests.get(
                f"{self.api_url}/recommend/detail/1?user_id=999&limit=5",
                timeout=5
            )
            if resp.status_code == 200:
                test_requests.append(("recommend", True, resp.json()))
            else:
                test_requests.append(("recommend", False, f"状态码: {resp.status_code}"))
        except Exception as e:
            test_requests.append(("recommend", False, str(e)))

        # 等待日志写入
        time.sleep(1)

        # 检查日志是否增长
        if self.exposure_log.exists():
            new_size = self.exposure_log.stat().st_size
            with open(self.exposure_log, 'r') as f:
                new_lines = len(f.readlines())

            growth = new_size - initial_size
            new_entries = new_lines - initial_lines

            if new_entries > 0:
                self.check(
                    "曝光日志写入",
                    True,
                    f"成功记录 {new_entries} 条曝光日志",
                    f"文件增长: {growth} bytes"
                )

                # 读取最后几条日志验证格式
                with open(self.exposure_log, 'r') as f:
                    lines = f.readlines()
                    last_entries = lines[-min(3, len(lines)):]

                valid_format = True
                for line in last_entries:
                    try:
                        entry = json.loads(line.strip())
                        required_fields = ['request_id', 'algorithm_version', 'items', 'timestamp']
                        if not all(field in entry for field in required_fields):
                            valid_format = False
                            break
                    except json.JSONDecodeError:
                        valid_format = False
                        break

                if valid_format:
                    self.check(
                        "曝光日志格式",
                        True,
                        "日志格式正确，包含所有必需字段",
                        f"示例: {last_entries[-1][:100]}..."
                    )
                else:
                    self.check(
                        "曝光日志格式",
                        False,
                        "日志格式不正确或缺少必需字段"
                    )

                return True
            else:
                self.check(
                    "曝光日志写入",
                    False,
                    "调用API后日志文件没有增长",
                    f"API调用结果: {test_requests}"
                )
                return False
        else:
            self.check(
                "曝光日志写入",
                False,
                "调用API后日志文件未创建",
                f"API调用结果: {test_requests}"
            )
            return False

    def verify_matomo_data(self) -> bool:
        """验证Matomo数据读取"""
        print("=" * 60)
        print("3. Matomo数据读取验证")
        print("=" * 60)

        matomo_dir = DATA_DIR / "matomo"

        if not matomo_dir.exists():
            self.check(
                "Matomo数据目录",
                False,
                f"Matomo数据目录不存在: {matomo_dir}",
                "需要先运行 pipeline.extract_load 抽取数据"
            )
            return False

        # 检查关键表
        tables = [
            "matomo_log_action.parquet",
            "matomo_log_link_visit_action.parquet",
            "matomo_log_conversion.parquet"
        ]

        all_exist = True
        table_info = []

        for table in tables:
            table_path = matomo_dir / table
            if table_path.exists():
                size = table_path.stat().st_size
                table_info.append(f"{table}: {size // 1024}KB")
            else:
                all_exist = False
                table_info.append(f"{table}: 缺失")

        if all_exist:
            self.check(
                "Matomo核心表",
                True,
                "所有核心Matomo表已抽取",
                ", ".join(table_info)
            )
        else:
            self.check(
                "Matomo核心表",
                False,
                "部分Matomo表缺失",
                ", ".join(table_info)
            )

        # 检查数据新鲜度
        action_path = matomo_dir / "matomo_log_link_visit_action.parquet"
        if action_path.exists():
            modified_time = datetime.fromtimestamp(action_path.stat().st_mtime)
            age_hours = (datetime.now() - modified_time).total_seconds() / 3600

            if age_hours < 48:
                self.check(
                    "Matomo数据新鲜度",
                    True,
                    f"数据较新，{age_hours:.1f}小时前更新",
                    f"最后更新: {modified_time}"
                )
            else:
                self.check(
                    "Matomo数据新鲜度",
                    False,
                    f"数据较旧，{age_hours:.1f}小时前更新",
                    "建议运行增量抽取更新数据"
                )

        return all_exist

    def verify_evaluation_pipeline(self) -> bool:
        """验证评估pipeline能否运行"""
        print("=" * 60)
        print("4. 评估Pipeline验证")
        print("=" * 60)

        # 检查是否有足够数据运行评估
        eval_dir = DATA_DIR / "evaluation"

        if not eval_dir.exists():
            self.check(
                "评估目录",
                False,
                f"评估目录不存在: {eval_dir}",
                "需要先运行一次完整pipeline"
            )
            return False

        # 检查评估输出文件
        output_files = {
            "summary.json": "总体评估指标",
            "dataset_metrics.csv": "数据集级别指标",
            "exposure_metrics.json": "曝光CTR/CVR指标"
        }

        files_exist = []
        for filename, desc in output_files.items():
            file_path = eval_dir / filename
            if file_path.exists():
                files_exist.append(f"{desc} ({filename})")

        if files_exist:
            self.check(
                "评估输出文件",
                True,
                f"找到 {len(files_exist)}/{len(output_files)} 个评估文件",
                ", ".join(files_exist)
            )

            # 读取summary看指标
            summary_path = eval_dir / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)

                    ctr = summary.get('exposures_ctr', 0)
                    cvr = summary.get('exposures_cvr', 0)
                    exposures = summary.get('exposures_total', 0)

                    self.check(
                        "CTR/CVR指标",
                        True,
                        f"已有评估指标: CTR={ctr:.4f}, CVR={cvr:.4f}",
                        f"总曝光数: {exposures}"
                    )
                except Exception as e:
                    self.check(
                        "CTR/CVR指标",
                        False,
                        f"读取评估指标失败: {e}"
                    )
        else:
            self.check(
                "评估输出文件",
                False,
                "未找到评估输出文件",
                "需要运行 pipeline.evaluate 生成评估报告"
            )

        return True

    def verify_data_structure(self) -> bool:
        """验证数据目录结构完整性"""
        print("=" * 60)
        print("5. 数据目录结构验证")
        print("=" * 60)

        required_dirs = {
            "business": "业务数据",
            "matomo": "Matomo行为数据",
            "processed": "处理后特征数据",
            "evaluation": "评估结果"
        }

        all_exist = True
        dir_status = []

        for dirname, desc in required_dirs.items():
            dir_path = DATA_DIR / dirname
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*.parquet"))) + len(list(dir_path.glob("*.json")))
                dir_status.append(f"{desc}: ✓ ({file_count} 文件)")
            else:
                all_exist = False
                dir_status.append(f"{desc}: ✗ 缺失")

        self.check(
            "数据目录结构",
            all_exist,
            "数据目录" + ("完整" if all_exist else "不完整"),
            "\n     ".join(dir_status)
        )

        return all_exist

    def generate_report(self):
        """生成验证报告"""
        print("\n" + "=" * 60)
        print("验证报告汇总")
        print("=" * 60)

        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed

        print(f"\n总检查项: {total}")
        print(f"通过: {passed} ✓")
        print(f"失败: {failed} ✗")
        print(f"通过率: {passed/total*100:.1f}%\n")

        if failed > 0:
            print("失败项详情：")
            for r in self.results:
                if not r['passed']:
                    print(f"  - {r['name']}: {r['message']}")
                    if r['details']:
                        print(f"    {r['details']}")

        # 保存报告
        report_path = DATA_DIR / "evaluation" / "tracking_verification.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "pass_rate": passed / total if total > 0 else 0
                },
                "checks": self.results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n完整报告已保存: {report_path}")

        return failed == 0

    def run_all_checks(self) -> bool:
        """运行所有验证"""
        print("\n推荐系统埋点验证工具")
        print(f"开始时间: {datetime.now()}")
        print(f"API地址: {self.api_url}")
        print(f"数据目录: {DATA_DIR}\n")

        # 按顺序执行检查
        self.verify_api_health()
        self.verify_exposure_logging()
        self.verify_matomo_data()
        self.verify_evaluation_pipeline()
        self.verify_data_structure()

        # 生成报告
        return self.generate_report()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="验证推荐系统埋点和数据收集")
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API服务地址 (默认: http://localhost:8000)"
    )

    args = parser.parse_args()

    verifier = TrackingVerifier(api_url=args.api_url)
    success = verifier.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
