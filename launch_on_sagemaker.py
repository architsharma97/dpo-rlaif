import argparse
import time
import os
import subprocess
import yaml
from datetime import datetime
from pathlib import Path

import boto3
from sagemaker.pytorch import PyTorch


NAME = "deam"
INSTANCE_MAPPER = {
    "p3": "ml.p3.16xlarge",
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
}


def run_command(command):
    subprocess.run(command, shell=True, check=True)


def get_image(user, instance_type, docker_dir, build_type=None, profile="poweruser", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    print(account)
    algorithm_name = f"{user}-{NAME}"
    dockerfile_base = docker_dir / "Dockerfile"
    dockerfile_update = docker_dir / "Dockerfile_update"
    
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if build_type is None:
        return fullname

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    if build_type == "full":
        print("Building container")
        commands = [
            # Log in to Sagemaker account to get image.
            f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
            f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} --build-arg INSTANCE_TYPE={instance_type} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
            (
                f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || "
                f"aws --region {region} ecr create-repository --repository-name {algorithm_name}"
            ),
        ]
    elif build_type == "update":
        print("Updating container")
        commands = [
            f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
            f"docker build --progress=plain -f {dockerfile_update} --build-arg BASE_DOCKER={algorithm_name} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
        ]

    else:
        raise ValueError(f"Unknown build_type: {build_type}")

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


def main(args=None):
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", choices=["full", "update"], help="Build image from scratch")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--user", required=True, help="User name")
    parser.add_argument("--cfg-path", required=True, help="Launch config")
    parser.add_argument("--role", default=None, help="Python version")
    parser.add_argument(
        "--job-suffix",
        default=None,
        help="Suffix to add to job name for easier identification",
    )
    parser.add_argument(
        "--docker-dir", type=Path, default=Path(__file__).parent
    )
    parser.add_argument("--entry-point", default="open_lm/main.py", help="Entry point to run")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default="default", help="AWS profile to use")

    # Instance args
    parser.add_argument("--instance-count", default=1, type=int, help="Number of instances")
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument("--spot-instance", action="store_true")
    parser.add_argument("--command", default="", choices=["python", "accelerate", ""])
    parser.add_argument("--model_uri", default="s3://tri-ml-datasets/scratch/archit.sharma/mistral7bsft0.1/policy.pt", type=str)
    parser.add_argument("--input_s3_path", type=str, default="s3://tri-ml-datasets/scratch/archit.sharma/mistral7bsft0.1/comparisons_gpt4/temp1.0_vs_chatgpt/annotations.json")
    args = parser.parse_args(args)

    # Need to rename setup.py to avoid Sagemaker treating this as a module and causing errors.
    setup_tmp_name = "./setup_renamed_for_sagemaker.py"
    print(f"Renaming ./setup.py to {setup_tmp_name}")
    try:
        os.rename("./setup.py", setup_tmp_name)
    except:
        print("Failed to rename setup file")
    try:
        main_after_setup_move(args)
    finally:
        try:
            os.rename(setup_tmp_name, "./setup.py")
            print("Renamed setup.py back")
        except:
            pass


def main_after_setup_move(args):
    params = yaml.safe_load(open(args.cfg_path, "r"))
    s3_sync_base_folder = params.get("remote-sync", None)
    folder_name = params.get("name", None)
    if s3_sync_base_folder is not None and folder_name is not None:
        if s3_sync_base_folder[-1] == "/":
            s3_sync_base_folder = s3_sync_base_folder[:-1]
        s3_sync_folder = f"{s3_sync_base_folder}/{folder_name}"
        print(f"Checking if {s3_sync_folder} exists")
        # Check if the folder exists and fail if it does
        s3 = boto3.client("s3")
        # Extract bucket from s3_sync_folder
        bucket = s3_sync_folder.split("/")[2]
        # Extract the rest of the path
        path = "/".join(s3_sync_folder.split("/")[3:])
        s3.head_object(Bucket=bucket, Key=path)
        if params.get("resume", None) is None:
            raise ValueError(f"{s3_sync_folder} already exists. Please rename this job or delete this bucket.")
    
    image = get_image(
        args.user,
        args.instance_type,
        docker_dir=args.docker_dir,
        build_type=args.build_type,
        profile=args.profile,
        region=args.region,
    )
    ##########
    # Create session and make sure of account and region
    ##########
    # provide a pre-existing role ARN as an alternative to creating a new role
    role = args.role
    if role is None:
        role = os.environ.get("SAGEMAKER_ROLE_ARN", None)

    assert role is not None, "Please provide a role to launch the job on SageMaker."

    role_name = role.split(["/"][-1])
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name[-1]}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account:{account}")

    session = boto3.session.Session()
    region = session.region_name
    print(f"AWS region:{region}")

    ##########
    # Configure the training
    ##########
    base_job_name = f"{args.user.replace('.', '-')}-{NAME}"

    checkpoint_local_path = "/opt/ml/checkpoints"

    with open(args.cfg_path, "r") as f:
        hyperparameters = yaml.safe_load(f)
        # Replace "True" with "" in hyperparameters
        hyperparameters = {k: v if v is not True else "" for k, v in hyperparameters.items()}

    hyperparameters['token'] = os.environ["HF_TOKEN"]
    def get_job_name(base, job_suffix=None):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"

        job_name = base
        if job_suffix:
            job_name = "_".join([job_name, job_suffix])

        job_name = "_".join([job_name, date_str])

        return job_name

    job_name = get_job_name(base_job_name, args.job_suffix)

    output_root = f"s3://tri-ml-sandbox-16011-us-east-1-datasets/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    print(f"Hyperparameters: {hyperparameters}")
    print(f"Running On: {args.instance_type}:{args.instance_count}")
    # entrypoint_dir = str(Path(args.entry_point).parent)
    # print(f"Source Directory: {entrypoint_dir}")

    estimator_args = {
        "source_dir": ".",
        "entry_point": args.entry_point,
        "base_job_name": base_job_name,
        "hyperparameters": hyperparameters,
        "role": role,
        "image_uri": image,
        "instance_count": int(args.instance_count),
        "instance_type": "local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        "use_spot_instances": True if args.spot_instance else False,
        "output_path": output_s3,
        "job_name": job_name,
        "checkpoint_s3_uri": None if args.local else f"{output_s3}/checkpoint",
        "checkpoint_local_path": checkpoint_local_path if not args.local else None,
        "code_location": output_s3,
        "max_run": 5 * 24 * 60 * 60,
        "max_wait": 5 * 24 * 60 * 60 if args.spot_instance else None,
        "input_mode": "FastFile",
        "keep_alive_period_in_seconds": 30 * 60 if not args.spot_instance else None,  # 30 minutes
        "volume_size": 1000,
        "model_uri": args.model_uri
    }

    if args.command == "python":
        estimator_args["command"] = ["python"]
    elif args.command == "accelerate":
        estimator_args["command"] = ["accelerate", "launch"]
    else:
        estimator_args["distribution"] = {"torch_distributed": {"enabled": True}}

    estimator = PyTorch(**estimator_args)
    estimator.fit(        
        inputs=args.input_s3_path
    )


if __name__ == "__main__":
    main()
