"""
测试 Sentry 集成

用于验证 Sentry SDK 是否正确配置并能成功发送事件。

使用方式:
    python scripts/test_sentry.py

预期结果:
    - 如果配置正确，会在 Sentry 项目中看到测试错误和消息
    - 脚本会输出 Sentry 事件 ID
"""
import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sentry_config import (
    init_sentry,
    set_user_context,
    set_request_context,
    set_recommendation_context,
    capture_exception_with_context,
    capture_message_with_context,
    add_breadcrumb,
)


def test_basic_initialization():
    """测试基本初始化"""
    print("=" * 60)
    print("测试 1: Sentry 基本初始化")
    print("=" * 60)

    success = init_sentry(
        service_name="test-script",
        enable_tracing=True,
        traces_sample_rate=1.0,  # 测试时使用 100% 采样
        environment="development",
    )

    if success:
        print("✓ Sentry 初始化成功")
    else:
        print("✗ Sentry 初始化失败（可能未配置 SENTRY_DSN）")
        return False

    return True


def test_simple_message():
    """测试简单消息发送"""
    print("\n" + "=" * 60)
    print("测试 2: 发送简单消息")
    print("=" * 60)

    try:
        import sentry_sdk

        event_id = sentry_sdk.capture_message("Sentry 集成测试消息", level="info")
        print(f"✓ 消息发送成功")
        print(f"  事件 ID: {event_id}")
        print(f"  请在 Sentry 项目中查看此事件")
        return True
    except Exception as e:
        print(f"✗ 消息发送失败: {e}")
        return False


def test_exception_capture():
    """测试异常捕获"""
    print("\n" + "=" * 60)
    print("测试 3: 捕获异常")
    print("=" * 60)

    try:
        # 故意触发一个异常
        result = 1 / 0
    except ZeroDivisionError as e:
        try:
            import sentry_sdk

            event_id = sentry_sdk.capture_exception(e)
            print(f"✓ 异常捕获成功")
            print(f"  事件 ID: {event_id}")
            print(f"  异常类型: {type(e).__name__}")
            return True
        except Exception as capture_error:
            print(f"✗ 异常捕获失败: {capture_error}")
            return False


def test_context_and_tags():
    """测试上下文和标签"""
    print("\n" + "=" * 60)
    print("测试 4: 设置上下文和标签")
    print("=" * 60)

    try:
        # 设置用户上下文
        set_user_context(user_id=12345)
        print("✓ 设置用户上下文")

        # 设置请求上下文
        set_request_context(
            request_id="test_req_123",
            endpoint="/test",
            dataset_id=999,
        )
        print("✓ 设置请求上下文")

        # 设置推荐上下文
        set_recommendation_context(
            algorithm_version="test_v1.0",
            variant="primary",
            experiment_variant="test_experiment",
            channel_weights={"behavior": 1.0, "content": 0.5},
        )
        print("✓ 设置推荐上下文")

        # 添加面包屑
        add_breadcrumb(
            message="测试面包屑",
            category="test",
            level="info",
            test_data="test_value",
        )
        print("✓ 添加面包屑")

        # 发送一条带上下文的消息
        capture_message_with_context(
            "带有完整上下文的测试消息",
            level="info",
            test_field="test_value",
            timestamp=time.time(),
        )
        print("✓ 发送带上下文的消息")

        return True
    except Exception as e:
        print(f"✗ 上下文设置失败: {e}")
        return False


def test_custom_exception():
    """测试自定义异常捕获"""
    print("\n" + "=" * 60)
    print("测试 5: 捕获自定义异常（带额外上下文）")
    print("=" * 60)

    try:
        # 模拟推荐系统异常
        class RecommendationError(Exception):
            pass

        try:
            raise RecommendationError("推荐引擎计算失败：模拟错误")
        except RecommendationError as e:
            capture_exception_with_context(
                e,
                level="error",
                fingerprint=["test", "recommendation_error"],
                dataset_id=888,
                user_id=777,
                algorithm_version="v1.0",
                error_context="测试环境下的模拟错误",
            )
            print(f"✓ 自定义异常捕获成功")
            print(f"  异常信息: {str(e)}")
            return True
    except Exception as e:
        print(f"✗ 自定义异常捕获失败: {e}")
        return False


def test_performance_tracking():
    """测试性能追踪"""
    print("\n" + "=" * 60)
    print("测试 6: 性能追踪")
    print("=" * 60)

    try:
        import sentry_sdk
        from app.sentry_config import start_transaction, start_span

        # 开始一个事务
        with start_transaction(name="test_recommendation", op="test") as transaction:
            print("✓ 开始事务")

            # 模拟召回阶段
            with start_span(op="recall", description="multi-channel recall"):
                time.sleep(0.1)
                print("  - 完成召回阶段 (100ms)")

            # 模拟排序阶段
            with start_span(op="ranking", description="LightGBM ranking"):
                time.sleep(0.05)
                print("  - 完成排序阶段 (50ms)")

            # 模拟缓存写入
            with start_span(op="cache.set", description="Redis cache write"):
                time.sleep(0.02)
                print("  - 完成缓存写入 (20ms)")

        print("✓ 性能追踪完成")
        return True
    except Exception as e:
        print(f"✗ 性能追踪失败: {e}")
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("Sentry 集成测试")
    print("=" * 60)

    results = []

    # 测试 1: 初始化
    if not test_basic_initialization():
        print("\n✗ Sentry 未初始化，跳过后续测试")
        print("请确保设置了 SENTRY_DSN 环境变量")
        return 1

    # 测试 2-6
    results.append(("简单消息", test_simple_message()))
    results.append(("异常捕获", test_exception_capture()))
    results.append(("上下文和标签", test_context_and_tags()))
    results.append(("自定义异常", test_custom_exception()))
    results.append(("性能追踪", test_performance_tracking()))

    # 等待事件发送
    print("\n" + "=" * 60)
    print("等待 Sentry 事件发送...")
    print("=" * 60)
    try:
        import sentry_sdk
        sentry_sdk.flush(timeout=5)
        print("✓ 事件发送完成")
    except Exception as e:
        print(f"⚠ 发送可能未完成: {e}")

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s} : {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")

    print("\n" + "=" * 60)
    print("下一步操作:")
    print("=" * 60)
    print("1. 登录 Sentry: https://trace.dianshudata.com")
    print("2. 进入项目 #11")
    print("3. 查看 Issues 和 Performance 页面")
    print("4. 确认能看到上述测试事件")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
