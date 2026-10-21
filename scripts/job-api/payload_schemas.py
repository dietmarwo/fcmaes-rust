"""Validates a worker's discovery keys with jsonschema (4.18 or later).

Checks every schema in `payload_schemas` and `input_schemas`, every example
payload and example input against them, that each payload schema rejects an
unknown top-level field, and the given payloads (the tutorials') against
the payload schema of their profile. Used by tutorials/job-api/run.py and
tutorials/surrogate/run.py; without jsonschema it checks nothing unless
JOB_API_REQUIRE_SCHEMA=1, which makes the package required. `run_examples`
runs every example through the runner as a client would build it.
"""
import copy
import hashlib
import json
import os
from importlib import metadata

try:
    import jsonschema
except ImportError:
    jsonschema = None

if jsonschema is None and os.environ.get('JOB_API_REQUIRE_SCHEMA') == '1':
    raise SystemExit('the jsonschema package is required (JOB_API_REQUIRE_SCHEMA=1)')
if jsonschema is not None:
    VERSION = tuple(int(part) for part in metadata.version('jsonschema').split('.')[:2])
    if VERSION < (4, 18):
        raise SystemExit(f'jsonschema >= 4.18 is required, found {metadata.version("jsonschema")}')


def _validator(schema):
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def check(describe, payloads=()):
    """Validates `describe` and `payloads`, (label, profile, payload) triples.

    Returns the number of validated payloads and inputs, 0 without jsonschema.
    """
    if jsonschema is None:
        return 0
    profiles = describe['profiles']
    schemas = describe['payload_schemas']
    assert sorted(schemas) == sorted(profiles), (sorted(schemas), profiles)
    payload = {profile: _validator(schema) for profile, schema in schemas.items()}
    inputs = {name: _validator(schema) for name, schema in describe.get('input_schemas', {}).items()}
    checked = 0
    for profile, validator in payload.items():
        assert schemas[profile].get('additionalProperties') is False, profile
    examples = describe['examples']
    assert {e['profile'] for e in examples} == set(profiles), 'a profile without an example'
    for example in examples:
        label = f"example {example['description']}"
        payload[example['profile']].validate(example['payload'])
        # An unknown top-level field is rejected, as the worker rejects it.
        extra = copy.deepcopy(example['payload'])
        extra['unknown_field'] = 1
        assert not payload[example['profile']].is_valid(extra), label
        checked += 1
        for item in example['inputs']:
            if 'format' in item:
                assert item['format'] in inputs, (label, item['format'])
                value = item['content'] if 'content' in item else json.loads(item['content_text'])
                inputs[item['format']].validate(value)
                checked += 1
    for label, profile, value in payloads:
        try:
            payload[profile].validate(value)
        except jsonschema.ValidationError as error:
            raise AssertionError(f'{label}: {error.message}') from error
        checked += 1
    return checked


def check_input(describe, format, value):
    """Validates an input file's JSON `value` against `input_schemas[format]`."""
    if jsonschema is None:
        return 0
    _validator(describe['input_schemas'][format]).validate(value)
    return 1


def run_examples(call, directory, worker, description, conforms=lambda schema, value: None):
    """Runs every example of `worker`'s `describe` through the runner `call`:
    its inputs written beside the job file with their SHA-256, a request key
    of its own. Each job must complete with a claim in every task."""
    runs = []
    for index, example in enumerate(description['examples']):
        job_dir = directory / worker / str(index)
        job_dir.mkdir(parents=True, exist_ok=True)
        inputs = []
        for item in example['inputs']:
            if 'content_text' in item:
                data = item['content_text'].encode()
            else:
                data = json.dumps(item['content']).encode()
            (job_dir / item['file']).write_bytes(data)
            inputs.append({'name': item['name'], 'file': item['file'],
                           'sha256': hashlib.sha256(data).hexdigest()})
        spec = {'schema_version': 'optimization.job/0.1',
                'request_key': f'discovery.{worker}.{index}.v1', 'worker': worker,
                'payload_schema': example['profile'], 'payload': example['payload'],
                'resources': example['resources'], 'limits': example['limits'], 'inputs': inputs}
        path = job_dir / 'job.json'
        path.write_text(json.dumps(spec, indent=2) + '\n')
        conforms('optimization.job-0.1.schema.json', spec)
        call('validate', '--spec', path)
        job = call('run', '--spec', path)
        result = call('result', job['directory'])
        label = f"{worker}: {example['description']}"
        assert job['state'] == 'completed' and result['results'], (label, result)
        assert all(r['outcome']['claim'] != 'none' for r in result['results']), (label, result)
        runs.append({'worker': worker, 'profile': example['profile'],
                     'description': example['description'], 'job_id': job['id']})
    return runs
