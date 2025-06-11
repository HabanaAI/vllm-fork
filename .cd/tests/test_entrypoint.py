# SPDX-License-Identifier: Apache-2.0
from entrypoint import Entrypoint


def test_generate_server_script(tmp_path):
    entry = Entrypoint()
    template = tmp_path / "template.sh"
    output = tmp_path / "output.sh"
    # Prepare a simple template
    template.write_text("#@VARS\nrun")
    entry.generate_server_script(str(template), str(output), {
        "FOO": "bar",
        "X": 1
    })
    content = output.read_text()
    assert "export FOO=bar" in content
    assert "export X=1" in content
    assert "run" in content
