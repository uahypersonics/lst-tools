from unittest.mock import patch, MagicMock
from lst_tools.utils.progress import progress, _RichCtx


class TestProgress:
    """Test cases for the progress utility."""

    def test_progress_returns_rich_ctx(self):
        """Test that progress() returns a _RichCtx."""
        ctx = progress(total=100, desc="Test")
        assert isinstance(ctx, _RichCtx)

    def test_progress_context_manager(self):
        """Test progress context manager lifecycle with mocked rich."""
        with patch("lst_tools.utils.progress.Progress") as mock_progress:
            mock_prog_instance = MagicMock()
            mock_progress.return_value = mock_prog_instance
            mock_task = MagicMock()
            mock_prog_instance.add_task.return_value = mock_task

            with progress(total=100, desc="Test", persist=True) as advance:
                assert callable(advance)
                advance(10)
                advance(20)

            # verify rich methods were called
            mock_prog_instance.start.assert_called_once()
            mock_prog_instance.add_task.assert_called_once_with("Test", total=100)
            assert mock_prog_instance.update.call_count == 2
            mock_prog_instance.stop.assert_called_once()

    def test_progress_with_description_alias(self):
        """Test that description parameter works as alias for desc."""
        with patch("lst_tools.utils.progress.Progress") as mock_progress:
            mock_prog_instance = MagicMock()
            mock_progress.return_value = mock_prog_instance
            mock_prog_instance.add_task.return_value = MagicMock()

            with progress(total=50, description="Alias Test") as advance:
                advance(5)

            mock_prog_instance.add_task.assert_called_once_with("Alias Test", total=50)

    def test_progress_extra_kwargs_ignored(self):
        """Test that extra kwargs are ignored without errors."""
        # should not raise
        ctx = progress(total=100, desc="Test", unknown_param=True, another=42)
        assert isinstance(ctx, _RichCtx)

    def test_progress_persist_false_sets_transient(self):
        """Test that persist=False passes transient=True to rich."""
        with patch("lst_tools.utils.progress.Progress") as mock_progress:
            mock_prog_instance = MagicMock()
            mock_progress.return_value = mock_prog_instance
            mock_prog_instance.add_task.return_value = MagicMock()

            with progress(total=10, desc="Transient", persist=False) as advance:
                advance(1)

            # transient should be True when persist is False
            call_kwargs = mock_progress.call_args
            assert call_kwargs.kwargs["transient"] is True
