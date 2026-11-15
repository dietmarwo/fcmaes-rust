#!/usr/bin/env python3
"""Regression tests for broken links, moved pages and default-branch checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_learning_paths import Sources, anchors, check_page, is_stub, links


class ParsingTests(unittest.TestCase):
    def test_anchors_ignore_fences_and_suffix_duplicates(self):
        self.assertEqual(anchors('# First `run`!\n## First run\n```sh\n# fake\n```\n'),
                         {'first-run', 'first-run-1'})

    def test_reference_links_and_inline_code_fences(self):
        self.assertEqual(links('[A](a.md#run) [B][b]\n[b]: b.md\n```\n[X](fake.md)\n```'),
                         ['a.md#run', 'b.md'])
        with self.assertRaisesRegex(ValueError, 'undefined'):
            links('[Missing][no]')

    def test_moved_stubs_without_explicit_marker(self):
        for text in ['# Tutorial\n\nThe tutorial is now part of the [guide](guide.md).',
                     '# Old page\n\nThis page has moved to the guide.',
                     '<!-- redirect stub -->']:
            self.assertTrue(is_stub(text))
        self.assertFalse(is_stub('# Actual guide\nWe moved a particle through the model.'))
        self.assertFalse(is_stub('# Actual guide\nThe manual describes how the model was moved.'))


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.parent = Path(self.directory.name)
        self.root = self.parent / 'fcmaes-rust'
        (self.root / 'docs').mkdir(parents=True)
        self.page = self.root / 'docs/learning-paths.md'
        self.sources = Sources(self.parent, self.root)
        self.sources.cache['fcmaes-rust'] = self.root, 'main'

    def test_missing_file_anchor_and_stub_fail_independently(self):
        (self.root / 'docs/real.md').write_text('# Real\n## Works\n')
        (self.root / 'docs/old.md').write_text('# Old\nThis page has moved to real.md.')
        self.page.write_text('# Tracks\n[A](real.md#works) [B](real.md#absent) '
                             '[C](old.md) [D](missing.md) [E](#tracks)')
        report = check_page(self.page, self.sources)
        self.assertFalse(report['passed'])
        self.assertEqual(report['links_checked'], 5)
        self.assertEqual(len(report['errors']), 3)
        self.assertIn('missing anchor', report['errors'][0]['error'])
        self.assertIn('moved-page stub', report['errors'][1]['error'])
        self.assertIn('missing target', report['errors'][2]['error'])

    def test_source_url_must_use_actual_default_branch(self):
        (self.root / 'README.md').write_text('# Read me\n')
        self.sources.check('https://github.com/dietmarwo/fcmaes-rust/blob/main/README.md#read-me',
                           self.page, '')
        with self.assertRaisesRegex(ValueError, 'default branch'):
            self.sources.check('https://github.com/dietmarwo/fcmaes-rust/blob/master/README.md',
                               self.page, '')

    def test_reject_path_escape_and_line_limit(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.sources.check('../../secrets.md', self.page, '')
        self.page.write_text('# Title\n' + '\n' * 250)
        self.assertFalse(check_page(self.page, self.sources)['passed'])

    def test_remote_fetch_is_cached_and_uses_default_head(self):
        remote = self.parent / 'remote'
        remote.mkdir()
        sources = Sources(self.parent, self.root, remote)
        def fake_command(argv, timeout=180):
            if argv[1] == 'clone':
                Path(argv[-1]).mkdir()
                return ''
            return 'origin/master'
        with patch('check_learning_paths.command', side_effect=fake_command) as call:
            self.assertEqual(sources.checkout('fast-cma-es'), (remote/'fast-cma-es', 'master'))
            sources.checkout('fast-cma-es')
            self.assertEqual(call.call_count, 2)

    def test_remote_mode_reads_this_repository_from_the_working_tree(self):
        # A pull request is checked against its own files, not against main.
        remote = self.parent / 'remote'
        remote.mkdir()
        sources = Sources(self.parent, self.root, remote)
        (self.root / 'docs/new.md').write_text('# New page\n')
        self.page.write_text('# Tracks\n[N](new.md#new-page) '
                             '[A](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/new.md) '
                             '[G](gone.md)')
        def fake_command(argv, timeout=180):
            if argv[1] == 'clone':
                raise AssertionError('the repository under test must not be cloned')
            if argv[1] == 'ls-remote':
                return 'ref: refs/heads/main\tHEAD\n0123abc\tHEAD'
            raise AssertionError(argv)
        with patch('check_learning_paths.command', side_effect=fake_command):
            report = check_page(self.page, sources)
            with self.assertRaisesRegex(ValueError, 'default branch'):
                sources.check('https://github.com/dietmarwo/fcmaes-rust/blob/master/docs/new.md',
                              self.page, '')
        self.assertEqual([e['link'] for e in report['errors']], ['gone.md'], report)


if __name__ == '__main__':
    unittest.main()
