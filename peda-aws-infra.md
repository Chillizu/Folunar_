# PEDA - AWS 基础设施

建於 2026-07-17，配额已批. 状态更新于 2026-07-17.

---

## 已就绪的资源

| 资源 | 名称 | Region |
|---|---|---|
| S3 桶 | `chillizu-peda-checkpoints` | us-east-1 |
| IAM 实例角色 | `PEDA-Spot-Role` | — |
| IAM 实例配置文件 | 同名 `PEDA-Spot-Role` | — |
| 预算告警 | `monthly-spend-alert` ($25/月, 三邮箱通知) | us-east-1 |
| 配额申请 | `Running On-Demand G and VT instances`: 0 → **8 vCPU** (工单 #178425847000763) | us-east-1 |

Role 权限范围：仅限于 `chillizu-peda-checkpoints` 桶的 Get/Put/List/Delete，无其他权限。

---

## 配额申请状态

- 2026-05-05: 4 vCPU 请求 → CASE_CLOSED (未跟进，自动关闭)
- 2026-07-17 03:21: 8 vCPU 请求 → CASE_OPENED (工单 #178425847000763)
- 2026-07-17 当天: 8 vCPU 请求 → **APPROVED** (回复详细用途后获批)
- 当前配额: **8 vCPU** (Running On-Demand G and VT instances)
- AWS 提示: 生效可能有最长 1 小时延迟, 批后稍等即可使用

查看工单: https://support.console.aws.amazon.com/support/home#/case/?displayId=178425847000763

> 2026-07-17 实测: 距离提交约几小时内获批，提交 appeal (附详细用途说明) 后 AWS 自动通过。


---

## 配额批了之后: 起一个 GPU Spot 实例

```bash
# 1. 找最新 Amazon Linux 2023 AMI (Deep Learning AMI 也可)
AMI=$(aws ssm get-parameters-by-path --path /aws/service/ami-amazon-linux-latest --region us-east-1 --query 'Parameters[?contains(Name,`al2023-ami`)].Value' --output text)

# 2. 启动 spot 实例, 挂 PEDA-Spot-Role
aws ec2 run-instances \
  --image-id $AMI \
  --instance-type g4dn.xlarge \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --iam-instance-profile Name=PEDA-Spot-Role \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=PEDA-Spot}]' \
  --region us-east-1

# 3. SSH 进去后挂 S3 桶
sudo dnf install s3fs-fuse
s3fs chillizu-peda-checkpoints /mnt/peda-checkpoints

# 4. 训练, checkpoint 直接写 /mnt/peda-checkpoints/
python train.py --checkpoint-dir /mnt/peda-checkpoints/run-001

# 5. 练完关掉实例 (不保留 EBS 卷, 零残留)
aws ec2 terminate-instances --instance-ids i-xxxxx
```

练完后实例消失, EBS 卷不保留(省 $0.13/月), checkpoints 在 S3 上存活。

---

## 邪修数据

| 实例 | 单价模式 | 5 分钟实验 | $100 能跑 |
|---|---|---|---|
| g4dn.xlarge spot | ~$0.26/hr | ~$0.022 | ~4500 次 |
| g5.xlarge spot | ~$0.52/hr | ~$0.043 | ~2300 次 |
| t4g.micro (Always Free) | 免费 / $7/月 | CPU 推理 可用 | 不花钱 |

---

## 未来可能

- SageMaker Training Job + Managed Spot: 比手搓 EC2 更省心, 每次实验自动起停, 秒级计费
- Step Functions: 把训练管线编排成自动 workflow, push 代码自动跑实验
- Lambda (CPU): 15 分钟内能跑完的小实验完全免费 (1M 请求/月 + 400K GB-s)
