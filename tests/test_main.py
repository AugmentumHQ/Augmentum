"""Tests for augmentum/main.py and augmentum/__main__.py."""

from __future__ import annotations

from unittest.mock import patch


class TestMain:
    """Verify main entry point."""

    def test_main_calls_uvicorn_run(self):
        with patch("augmentum.main.uvicorn") as mock_uvicorn:
            from augmentum.main import main
            main()
            mock_uvicorn.run.assert_called_once()

    def test_main_passes_host_and_port(self):
        with patch("augmentum.main.uvicorn") as mock_uvicorn:
            from augmentum.main import main
            main()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs.kwargs.get("host") == "0.0.0.0" or call_kwargs[1].get("host") == "0.0.0.0"

    def test_main_uses_factory_mode(self):
        with patch("augmentum.main.uvicorn") as mock_uvicorn:
            from augmentum.main import main
            main()
            call_kwargs = mock_uvicorn.run.call_args
            # factory=True should be passed
            assert call_kwargs.kwargs.get("factory") is True or call_kwargs[1].get("factory") is True

    def test_main_passes_app_string(self):
        with patch("augmentum.main.uvicorn") as mock_uvicorn:
            from augmentum.main import main
            main()
            call_args = mock_uvicorn.run.call_args
            assert call_args[0][0] == "augmentum.proxy.server:create_app"

    def test_main_sets_up_logging(self):
        with patch("augmentum.main.uvicorn"):
            with patch("augmentum.main.setup_logging") as mock_setup:
                from augmentum.main import main
                main()
                mock_setup.assert_called_once()


class TestDunderMain:
    """Verify __main__.py imports cleanly."""

    def test_dunder_main_imports_with_mock(self):
        # __main__.py calls main() at import time, so we must mock uvicorn
        with patch("augmentum.main.uvicorn"):
            import importlib

            import augmentum.__main__  # noqa: F401
            importlib.reload(augmentum.__main__)

    def test_main_function_importable(self):
        from augmentum.main import main
        assert callable(main)

    def test_main_module_has_main(self):
        import augmentum.main as m
        assert hasattr(m, "main")


class TestMainIntegration:
    """Verify main integrates settings correctly."""

    def test_main_uses_settings_port(self):
        with patch("augmentum.main.uvicorn") as mock_uvicorn:
            with patch("augmentum.main.settings") as mock_settings:
                mock_settings.log_level = "INFO"
                mock_settings.host = "127.0.0.1"
                mock_settings.port = 9999
                from augmentum.main import main
                main()
                call_kwargs = mock_uvicorn.run.call_args
                assert call_kwargs.kwargs.get("port") == 9999 or call_kwargs[1].get("port") == 9999

    def test_main_uses_settings_log_level(self):
        with patch("augmentum.main.uvicorn") as mock_uvicorn:
            with patch("augmentum.main.settings") as mock_settings:
                mock_settings.log_level = "DEBUG"
                mock_settings.host = "0.0.0.0"
                mock_settings.port = 6100
                from augmentum.main import main
                main()
                call_kwargs = mock_uvicorn.run.call_args
                assert call_kwargs.kwargs.get("log_level") == "debug" or call_kwargs[1].get("log_level") == "debug"
