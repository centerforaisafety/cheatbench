"""Large image-bearing events must not stop draining the child process."""
import asyncio
import sys
from core.trial import _stream_lines


def test_lines_larger_than_stream_limit_preserve_utf8_and_final_fragment():
    async def check():
        reader = asyncio.StreamReader(limit=32)
        expected = [("\x1eRHMSG " + "é" * 70000 + "\n").encode(), b"next\n", b"final"]
        data = b"".join(expected)
        async def feed():
            for start in range(0, len(data), 17):
                reader.feed_data(data[start:start+17])
                await asyncio.sleep(0)
            reader.feed_eof()
        feeder = asyncio.create_task(feed())
        got = [line async for line in _stream_lines(reader)]
        await feeder
        assert got == expected
    asyncio.run(check())


def test_subprocess_large_event_does_not_deadlock_stdout():
    async def check():
        proc = await asyncio.create_subprocess_exec(sys.executable, "-c",
            "import sys;sys.stderr.write('x'*200000+'\\n');sys.stderr.flush();print('DONE')",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=64)
        async def drain():
            return [line async for line in _stream_lines(proc.stderr)]
        stdout, lines = await asyncio.wait_for(asyncio.gather(proc.stdout.read(), drain()), 5)
        assert await proc.wait() == 0
        assert stdout == b'DONE\n'
        assert lines == [b'x'*200000+b'\n']
    asyncio.run(check())
