import unittest

from .common import FS_OUT, hms_to_seconds
from .segment_rules import SeizureEvent, build_segments_for_file


def event(number, onset_min, offset_min, file='x.edf'):
    return SeizureEvent(str(number), file, onset_min * 60, offset_min * 60, '', '')


def labels(segments, prefix):
    return [s for s in segments if s['Label'].startswith(prefix)]


class RetainedBlockRules(unittest.TestCase):
    def test_seizure_free_recording_is_all_interictal(self):
        segments, counter = build_segments_for_file('x.edf', 2 * 3600, [], 0)
        self.assertEqual(counter, 0)
        self.assertEqual([s['Span'] for s in labels(segments, 'Inter')], [[0, 2 * 3600 * FS_OUT]])

    def test_short_preamble_retains_valid_minute(self):
        segments, _ = build_segments_for_file('x.edf', 40 * 60, [event(1, 20, 21)], 0)
        self.assertEqual(len(labels(segments, 'Pre')), 1)
        self.assertEqual(labels(segments, 'Pre')[0]['MinuteBlocks'], 19)
        self.assertTrue(labels(segments, 'Onset')[0]['PreictalEligible'])
        self.assertFalse(labels(segments, 'Inter'))

    def test_all_seizures_guard_interictal_even_when_pre_ineligible(self):
        segments, _ = build_segments_for_file('x.edf', 3 * 3600,
                                              [event(1, 90, 91), event(2, 110, 111)], 0)
        self.assertEqual(len(labels(segments, 'Pre')), 1)
        self.assertFalse(labels(segments, 'Onset')[1]['PreictalEligible'])
        self.assertEqual([s['Span'] for s in labels(segments, 'Inter')],
                         [[0, 30 * 60 * FS_OUT], [171 * 60 * FS_OUT, 180 * 60 * FS_OUT]])

    def test_partial_fragments_do_not_combine_into_minute(self):
        segments, _ = build_segments_for_file(
            'x.edf', 10 * 60, [event(1, 3, 4)], 0,
            invalid_spans=[(0, 90 * FS_OUT)])
        self.assertFalse(labels(segments, 'Pre'))
        self.assertFalse(labels(segments, 'Onset')[0]['PreictalEligible'])

    def test_context_seizure_guards_pre_and_interictal(self):
        segments, _ = build_segments_for_file(
            'x.edf', 2 * 3600, [event(1, 30, 31)], 0,
            context_events=[event('prior', -20, -19, 'previous.edf')])
        self.assertFalse(labels(segments, 'Pre'))
        self.assertEqual(len(labels(segments, 'Onset')), 1)

    def test_file_boundary_clips_an_ongoing_seizure(self):
        segments, _ = build_segments_for_file('x.edf', 10 * 60, [event(1, 8, 12)], 0)
        onset = labels(segments, 'Onset')[0]
        self.assertEqual(onset['Span'], [8 * 60 * FS_OUT, 10 * 60 * FS_OUT])
        self.assertTrue(onset['OffsetClippedByFileBoundary'])

    def test_release_clock_notes_and_ambiguity(self):
        self.assertEqual(hms_to_seconds('1 6.49.25'), 16 * 3600 + 49 * 60 + 25)
        self.assertEqual(hms_to_seconds('15.18.26 (CLINICAL ONSET)'), 15 * 3600 + 18 * 60 + 26)
        with self.assertRaisesRegex(ValueError, 'annotation override'):
            hms_to_seconds('15.43.53 (CLINICAL); 15.43.59 (ELECTRIC)')


if __name__ == '__main__':
    unittest.main()
