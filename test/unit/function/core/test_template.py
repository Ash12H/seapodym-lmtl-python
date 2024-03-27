from seapopym.function.core.template import generate_template


class TestTemplate:
    def test_template_no_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        res = generate_template(
            state=state_preprod_fg4_t4d_y1_x1_z3,
            dims=["Y", "X", "Z"],
            attributs={"units": "meter"},
            name="test",
        )
        assert res.chunks is None

    def test_template_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        res = generate_template(
            state=state_preprod_fg4_t4d_y1_x1_z3,
            dims=["Y", "X", "Z"],
            attributs={"units": "meter"},
            name="test",
            chunk={"Y": 1, "X": 1, "Z": 1},
        )
        assert res.chunks == ((1,), (1,), (3,))  # dim y = 1, dim x = 1, dim z = 3
