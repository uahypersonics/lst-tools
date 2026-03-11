import pytest
from unittest.mock import patch, MagicMock
from lst_tools.utils.progress import progress, _HAVE_RICH, _HAVE_TQDM


class TestProgress:
    """Test cases for the progress utility."""

    def test_progress_no_libraries(self):
        """Test progress when no external libraries are available."""
        with (
            patch("lst_tools.utils.progress._HAVE_RICH", False),
            patch("lst_tools.utils.progress._HAVE_TQDM", False),
        ):
            # Import the classes after patching
            from lst_tools.utils.progress import _NoopCtx

            ctx = progress(total=100, desc="Test")
            assert isinstance(ctx, _NoopCtx)

            # Test that noop context works without errors
            with ctx as advance:
                advance(10)
                advance(20)

    @pytest.mark.skipif(not _HAVE_RICH, reason="rich not available")
    def test_progress_rich_available(self):
        """Test progress with rich available."""
        with (
            patch("lst_tools.utils.progress._HAVE_RICH", True),
            patch("lst_tools.utils.progress._HAVE_TQDM", False),
        ):
            # Mock the rich Progress class
            with patch("lst_tools.utils.progress.Progress") as mock_progress:
                mock_prog_instance = MagicMock()
                mock_progress.return_value = mock_prog_instance
                mock_task = MagicMock()
                mock_prog_instance.add_task.return_value = mock_task

                ctx = progress(total=100, desc="Rich Test", persist=True)
                from lst_tools.utils.progress import _RichCtx

                assert isinstance(ctx, _RichCtx)

                with ctx as advance:
                    advance(10)
                    advance(20)

                # Verify rich methods were called
                mock_prog_instance.start.assert_called_once()
                mock_prog_instance.add_task.assert_called_once_with(
                    "Rich Test", total=100
                )
                mock_prog_instance.update.assert_called()
                mock_prog_instance.stop.assert_called_once()

    @pytest.mark.skipif(not _HAVE_TQDM, reason="tqdm not available")
    def test_progress_tqdm_available(self):
        """Test progress with tqdm available."""
        with (
            patch("lst_tools.utils.progress._HAVE_RICH", False),
            patch("lst_tools.utils.progress._HAVE_TQDM", True),
        ):
            # Mock the tqdm class
            with patch("lst_tools.utils.progress.tqdm") as mock_tqdm:
                mock_tqdm_instance = MagicMock()
                mock_tqdm.return_value = mock_tqdm_instance

                ctx = progress(total=100, desc="TQDM Test", persist=False)
                from lst_tools.utils.progress import _TqdmCtx

                assert isinstance(ctx, _TqdmCtx)

                with ctx as advance:
                    advance(10)
                    advance(20)

                # Verify tqdm methods were called
                mock_tqdm.assert_called_once_with(
                    total=100, desc="TQDM Test", leave=False
                )
                mock_tqdm_instance.update.assert_called()
                mock_tqdm_instance.close.assert_called_once()

    def test_progress_precedence_rich_over_tqdm(self):
        """Test that rich is preferred over tqdm when both are available."""
        with (
            patch("lst_tools.utils.progress._HAVE_RICH", True),
            patch("lst_tools.utils.progress._HAVE_TQDM", True),
        ):
            # Mock rich Progress class
            with patch("lst_tools.utils.progress.Progress") as mock_progress:
                mock_prog_instance = MagicMock()
                mock_progress.return_value = mock_prog_instance
                mock_task = MagicMock()
                mock_prog_instance.add_task.return_value = mock_task

                ctx = progress(total=50, desc="Precedence Test")
                from lst_tools.utils.progress import _RichCtx

                assert isinstance(ctx, _RichCtx)

    def test_progress_with_description_alias(self):
        """Test that description parameter works as alias for desc."""
        with (
            patch("lst_tools.utils.progress._HAVE_RICH", False),
            patch("lst_tools.utils.progress._HAVE_TQDM", False),
        ):
            from lst_tools.utils.progress import _NoopCtx

            # Test using description parameter
            ctx1 = progress(total=100, description="Test Description")
            assert isinstance(ctx1, _NoopCtx)

            # Test using desc parameter
            ctx2 = progress(total=100, desc="Test Desc")
            assert isinstance(ctx2, _NoopCtx)

    def test_progress_extra_kwargs_ignored(self):
        """Test that extra kwargs are ignored without errors."""
        with (
            patch("lst_tools.utils.progress._HAVE_RICH", False),
            patch("lst_tools.utils.progress._HAVE_TQDM", False),
        ):
            from lst_tools.utils.progress import _NoopCtx

            # Should not raise an error even with unknown kwargs
            ctx = progress(total=100, desc="Test", unknown_param=True, another_param=42)
            assert isinstance(ctx, _NoopCtx)

    def test_progress_persist_parameter(self):
        """Test that persist parameter is handled correctly."""
        with (
            patch("lst_tools.utils.progress._HAVE_RICH", False),
            patch("lst_tools.utils.progress._HAVE_TQDM", False),
        ):
            from lst_tools.utils.progress import _NoopCtx

            # Test with persist=True
            ctx1 = progress(total=100, desc="Persist Test", persist=True)
            assert isinstance(ctx1, _NoopCtx)

            # Test with persist=False
            ctx2 = progress(total=100, desc="No Persist Test", persist=False)
            assert isinstance(ctx2, _NoopCtx)

    def test_rich_ctx_context_manager(self):
        """Test _RichCtx context manager behavior."""
        with patch("lst_tools.utils.progress._HAVE_RICH", True):
            with patch("lst_tools.utils.progress.Progress") as mock_progress:
                mock_prog_instance = MagicMock()
                mock_progress.return_value = mock_prog_instance
                mock_task = MagicMock()
                mock_prog_instance.add_task.return_value = mock_task

                from lst_tools.utils.progress import _RichCtx

                ctx = _RichCtx(total=100, desc="Rich Ctx Test", persist=True)

                with ctx as advance:
                    assert callable(advance)
                    advance(10)

                # Verify cleanup
                mock_prog_instance.stop.assert_called_once()

    def test_tqdm_ctx_context_manager(self):
        """Test _TqdmCtx context manager behavior."""
        with (
            patch("lst_tools.utils.progress._HAVE_TQDM", True),
            patch("lst_tools.utils.progress._HAVE_RICH", False),
        ):
            with patch("lst_tools.utils.progress.tqdm") as mock_tqdm:
                mock_tqdm_instance = MagicMock()
                mock_tqdm.return_value = mock_tqdm_instance

                from lst_tools.utils.progress import _TqdmCtx

                ctx = _TqdmCtx(total=100, desc="TQDM Ctx Test", persist=False)

                with ctx as advance:
                    assert callable(advance)
                    advance(10)

                # Verify cleanup
                mock_tqdm_instance.close.assert_called_once()

    def test_noop_ctx_context_manager(self):
        """Test _NoopCtx context manager behavior."""
        from lst_tools.utils.progress import _NoopCtx

        ctx = _NoopCtx(total=100, desc="Noop Test", persist=True)

        with ctx as advance:
            assert callable(advance)
            # Should not raise any errors
            advance(10)
            advance(20)
