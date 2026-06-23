#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务连接测试脚本
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_milvus():
    """测试 Milvus 连接"""
    print("测试 Milvus 连接...")
    try:
        from pymilvus import connections, utility
        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_URL", "http://localhost:19530")
        )
        version = utility.get_server_version()
        print(f"  ✓ Milvus 连接成功，版本: {version}")
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"  ✗ Milvus 连接失败: {e}")
        return False

def test_neo4j():
    """测试 Neo4j 连接"""
    print("测试 Neo4j 连接...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            print(f"  ✓ Neo4j 连接成功，测试查询返回: {record['num']}")
        driver.close()
        return True
    except Exception as e:
        print(f"  ✗ Neo4j 连接失败: {e}")
        return False

def test_mongodb():
    """测试 MongoDB 连接"""
    print("测试 MongoDB 连接...")
    try:
        from pymongo import MongoClient
        client = MongoClient(
            os.getenv("MONGO_URL", "mongodb://localhost:27017"),
            serverSelectionTimeoutMS=5000
        )
        # 触发实际连接
        client.admin.command('ping')
        db_names = client.list_database_names()
        print(f"  ✓ MongoDB 连接成功，数据库列表: {db_names}")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ MongoDB 连接失败: {e}")
        return False

def test_minio():
    """测试 MinIO 连接"""
    print("测试 MinIO 连接...")
    try:
        from minio import Minio
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False
        )
        buckets = client.list_buckets()
        bucket_names = [b.name for b in buckets]
        print(f"  ✓ MinIO 连接成功，存储桶: {bucket_names}")
        return True
    except Exception as e:
        print(f"  ✗ MinIO 连接失败: {e}")
        return False

def main():
    print("=" * 50)
    print("掌柜问数 - 服务连接测试")
    print("=" * 50)

    results = {
        "Milvus": test_milvus(),
        "Neo4j": test_neo4j(),
        "MongoDB": test_mongodb(),
        "MinIO": test_minio(),
    }

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    all_passed = True
    for service, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {service}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("所有服务连接正常！")
    else:
        print("存在服务连接失败，请检查配置。")

    return all_passed

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)