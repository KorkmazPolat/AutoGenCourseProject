INFO:     Application startup complete.
INFO:     127.0.0.1:50607 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:50607 - "GET /login HTTP/1.1" 303 See Other
INFO:     127.0.0.1:50607 - "GET /dashboard HTTP/1.1" 200 OK
2026-07-25 18:14:50,299 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2026-07-25 18:14:50,303 INFO sqlalchemy.engine.Engine INSERT INTO courses (title, description, learning_outcomes, thumbnail_url, is_published, user_id, course_type) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id, created_at, updated_at
2026-07-25 18:14:50,303 INFO sqlalchemy.engine.Engine [generated in 0.00075s] ('New Video Course', 'Draft course created via builder.', '[]', None, 0, 1, 'course')
2026-07-25 18:14:50,307 INFO sqlalchemy.engine.Engine COMMIT
2026-07-25 18:14:50,322 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2026-07-25 18:14:50,326 INFO sqlalchemy.engine.Engine SELECT courses.id, courses.title, courses.description, courses.learning_outcomes, courses.thumbnail_url, courses.is_published, courses.created_at, courses.updated_at, courses.user_id, courses.course_type 
FROM courses 
WHERE courses.id = ?
2026-07-25 18:14:50,326 INFO sqlalchemy.engine.Engine [generated in 0.00033s] (8,)
INFO:     127.0.0.1:50607 - "GET /course-builder HTTP/1.1" 303 See Other
2026-07-25 18:14:50,329 INFO sqlalchemy.engine.Engine ROLLBACK
2026-07-25 18:14:50,337 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2026-07-25 18:14:50,339 INFO sqlalchemy.engine.Engine SELECT courses.id, courses.title, courses.description, courses.learning_outcomes, courses.thumbnail_url, courses.is_published, courses.created_at, courses.updated_at, courses.user_id, courses.course_type 
FROM courses 
WHERE courses.id = ?
2026-07-25 18:14:50,339 INFO sqlalchemy.engine.Engine [generated in 0.00039s] (8,)
2026-07-25 18:14:50,342 INFO sqlalchemy.engine.Engine SELECT modules.id, modules.course_id, modules.title, modules.summary, modules.order_index 
FROM modules 
WHERE modules.course_id = ? ORDER BY modules.order_index
2026-07-25 18:14:50,343 INFO sqlalchemy.engine.Engine [generated in 0.00049s] (8,)
INFO:     127.0.0.1:50607 - "GET /course-builder/8 HTTP/1.1" 200 OK
2026-07-25 18:14:50,352 INFO sqlalchemy.engine.Engine ROLLBACK
2026-07-25 18:15:33,308 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2026-07-25 18:15:33,310 INFO sqlalchemy.engine.Engine INSERT INTO courses (title, description, learning_outcomes, thumbnail_url, is_published, user_id, course_type) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id, created_at, updated_at
2026-07-25 18:15:33,310 INFO sqlalchemy.engine.Engine [generated in 0.00063s] ('My Designed Course', 'Manually designed course structure.', '["Manual Design"]', None, 0, 1, 'course')
2026-07-25 18:15:33,315 INFO sqlalchemy.engine.Engine INSERT INTO modules (course_id, title, summary, order_index) VALUES (?, ?, ?, ?)
2026-07-25 18:15:33,315 INFO sqlalchemy.engine.Engine [generated in 0.00045s] (9, 'New Module', 'Module 1', 0)
2026-07-25 18:15:33,323 INFO sqlalchemy.engine.Engine INSERT INTO lessons (module_id, title, content, order_index, duration_minutes) VALUES (?, ?, ?, ?, ?)
2026-07-25 18:15:33,324 INFO sqlalchemy.engine.Engine [generated in 0.00098s] (5, 'girişimcilik', 'bir girişimci için yatırıcmcı sunumu hazırlığı', 0, 3)
2026-07-25 18:15:33,327 INFO sqlalchemy.engine.Engine INSERT INTO lesson_assets (lesson_id, asset_type, content, file_path) VALUES (?, ?, ?, ?) RETURNING id, created_at
2026-07-25 18:15:33,328 INFO sqlalchemy.engine.Engine [generated in 0.00044s] (5, 'video', '{"status": "pending_generation", "type": "video"}', None)
2026-07-25 18:15:33,331 INFO sqlalchemy.engine.Engine COMMIT
INFO:     127.0.0.1:58684 - "POST /save-course-design HTTP/1.1" 200 OK
2026-07-25 18:15:49,162 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2026-07-25 18:15:49,163 INFO sqlalchemy.engine.Engine SELECT courses.id, courses.title, courses.description, courses.learning_outcomes, courses.thumbnail_url, courses.is_published, courses.created_at, courses.updated_at, courses.user_id, courses.course_type 
FROM courses 
WHERE courses.id = ?
2026-07-25 18:15:49,163 INFO sqlalchemy.engine.Engine [cached since 58.82s ago] (9,)
2026-07-25 18:15:49,165 INFO sqlalchemy.engine.Engine SELECT modules.id, modules.course_id, modules.title, modules.summary, modules.order_index 
FROM modules 
WHERE modules.course_id = ?
2026-07-25 18:15:49,165 INFO sqlalchemy.engine.Engine [generated in 0.00021s] (9,)
2026-07-25 18:15:49,167 INFO sqlalchemy.engine.Engine INSERT INTO modules (course_id, title, summary, order_index) VALUES (?, ?, ?, ?)
2026-07-25 18:15:49,167 INFO sqlalchemy.engine.Engine [cached since 15.85s ago] (9, 'New Module', 'Module 1', 0)
2026-07-25 18:15:49,172 INFO sqlalchemy.engine.Engine SELECT lessons.id, lessons.module_id, lessons.title, lessons.content, lessons.order_index, lessons.duration_minutes 
FROM lessons 
WHERE lessons.module_id = ?
2026-07-25 18:15:49,173 INFO sqlalchemy.engine.Engine [generated in 0.00032s] (6,)
2026-07-25 18:15:49,174 INFO sqlalchemy.engine.Engine INSERT INTO lessons (module_id, title, content, order_index, duration_minutes) VALUES (?, ?, ?, ?, ?)
2026-07-25 18:15:49,174 INFO sqlalchemy.engine.Engine [cached since 15.85s ago] (6, 'girişimcilik', 'bir girişimci için yatırıcmcı sunumu hazırlığı', 0, 3)
2026-07-25 18:15:49,178 INFO sqlalchemy.engine.Engine INSERT INTO lesson_assets (lesson_id, asset_type, content, file_path) VALUES (?, ?, ?, ?) RETURNING id, created_at
2026-07-25 18:15:49,178 INFO sqlalchemy.engine.Engine [cached since 15.85s ago] (6, 'video', '{"status": "pending"}', None)
2026-07-25 18:15:49,180 INFO sqlalchemy.engine.Engine SELECT lessons.id AS lessons_id, lessons.module_id AS lessons_module_id, lessons.title AS lessons_title, lessons.content AS lessons_content, lessons.order_index AS lessons_order_index, lessons.duration_minutes AS lessons_duration_minutes 
FROM lessons 
WHERE ? = lessons.module_id
2026-07-25 18:15:49,180 INFO sqlalchemy.engine.Engine [generated in 0.00022s] (5,)
2026-07-25 18:15:49,184 INFO sqlalchemy.engine.Engine SELECT lesson_assets.id AS lesson_assets_id, lesson_assets.lesson_id AS lesson_assets_lesson_id, lesson_assets.asset_type AS lesson_assets_asset_type, lesson_assets.content AS lesson_assets_content, lesson_assets.file_path AS lesson_assets_file_path, lesson_assets.created_at AS lesson_assets_created_at 
FROM lesson_assets 
WHERE ? = lesson_assets.lesson_id
2026-07-25 18:15:49,184 INFO sqlalchemy.engine.Engine [generated in 0.00031s] (5,)
2026-07-25 18:15:49,187 INFO sqlalchemy.engine.Engine SELECT quiz_attempts.id AS quiz_attempts_id, quiz_attempts.user_id AS quiz_attempts_user_id, quiz_attempts.asset_id AS quiz_attempts_asset_id, quiz_attempts.score AS quiz_attempts_score, quiz_attempts.max_score AS quiz_attempts_max_score, quiz_attempts.attempted_at AS quiz_attempts_attempted_at 
FROM quiz_attempts 
WHERE ? = quiz_attempts.asset_id
2026-07-25 18:15:49,187 INFO sqlalchemy.engine.Engine [generated in 0.00029s] (7,)
2026-07-25 18:15:49,190 INFO sqlalchemy.engine.Engine SELECT lesson_completions.id AS lesson_completions_id, lesson_completions.user_id AS lesson_completions_user_id, lesson_completions.lesson_id AS lesson_completions_lesson_id, lesson_completions.completed_at AS lesson_completions_completed_at 
FROM lesson_completions 
WHERE ? = lesson_completions.lesson_id
2026-07-25 18:15:49,190 INFO sqlalchemy.engine.Engine [generated in 0.00032s] (5,)
2026-07-25 18:15:49,192 INFO sqlalchemy.engine.Engine DELETE FROM lesson_assets WHERE lesson_assets.id = ?
2026-07-25 18:15:49,193 INFO sqlalchemy.engine.Engine [generated in 0.00020s] (7,)
2026-07-25 18:15:49,194 INFO sqlalchemy.engine.Engine DELETE FROM lessons WHERE lessons.id = ?
2026-07-25 18:15:49,194 INFO sqlalchemy.engine.Engine [generated in 0.00016s] (5,)
2026-07-25 18:15:49,195 INFO sqlalchemy.engine.Engine DELETE FROM modules WHERE modules.id = ?
2026-07-25 18:15:49,195 INFO sqlalchemy.engine.Engine [generated in 0.00018s] (5,)
2026-07-25 18:15:49,196 INFO sqlalchemy.engine.Engine COMMIT
INFO:     127.0.0.1:61845 - "POST /update-course-design/9 HTTP/1.1" 200 OK
INFO:     127.0.0.1:61845 - "POST /generate-course-content/9?video_engine=gemini HTTP/1.1" 200 OK
2026-07-25 18:15:51,727 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2026-07-25 18:15:51,728 INFO sqlalchemy.engine.Engine SELECT courses.id, courses.title, courses.description, courses.learning_outcomes, courses.thumbnail_url, courses.is_published, courses.created_at, courses.updated_at, courses.user_id, courses.course_type 
FROM courses 
WHERE courses.id = ?
2026-07-25 18:15:51,728 INFO sqlalchemy.engine.Engine [cached since 61.39s ago] (9,)
2026-07-25 18:15:51,730 INFO sqlalchemy.engine.Engine SELECT modules.id, modules.course_id, modules.title, modules.summary, modules.order_index 
FROM modules 
WHERE modules.course_id = ? ORDER BY modules.order_index
2026-07-25 18:15:51,731 INFO sqlalchemy.engine.Engine [cached since 61.39s ago] (9,)
2026-07-25 18:15:55,968 INFO sqlalchemy.engine.Engine SELECT count(lessons.id) AS count_1 
FROM lessons 
WHERE lessons.module_id = ?
2026-07-25 18:15:55,968 INFO sqlalchemy.engine.Engine [generated in 0.00044s] (6,)
2026-07-25 18:15:55,971 INFO sqlalchemy.engine.Engine SELECT lessons.id, lessons.module_id, lessons.title, lessons.content, lessons.order_index, lessons.duration_minutes 
FROM lessons 
WHERE lessons.module_id = ? ORDER BY lessons.order_index
2026-07-25 18:15:55,971 INFO sqlalchemy.engine.Engine [generated in 0.00049s] (6,)
2026-07-25 18:15:55,974 INFO sqlalchemy.engine.Engine SELECT lesson_assets.id, lesson_assets.lesson_id, lesson_assets.asset_type, lesson_assets.content, lesson_assets.file_path, lesson_assets.created_at 
FROM lesson_assets 
WHERE lesson_assets.lesson_id = ?
2026-07-25 18:15:55,974 INFO sqlalchemy.engine.Engine [generated in 0.00040s] (6,)
DEBUG: Generating video content for lesson 6 with 3 mins...
SUCCESS: Used Gemini for Video Design
Task exception was never retrieved
future: <Task finished name='Task-19' coro=<Connection.run() done, defined at C:\Users\mertk\OneDrive\Belgeler\GitHub\AutoGenCourseProject\.venv\Lib\site-packages\playwright\_impl\_connection.py:305> exception=NotImplementedError()>
Traceback (most recent call last):
  File "C:\Users\mertk\OneDrive\Belgeler\GitHub\AutoGenCourseProject\.venv\Lib\site-packages\playwright\_impl\_connection.py", line 312, in run
    await self._transport.connect()
  File "C:\Users\mertk\OneDrive\Belgeler\GitHub\AutoGenCourseProject\.venv\Lib\site-packages\playwright\_impl\_transport.py", line 133, in connect
    raise exc
  File "C:\Users\mertk\OneDrive\Belgeler\GitHub\AutoGenCourseProject\.venv\Lib\site-packages\playwright\_impl\_transport.py", line 120, in connect
    self._proc = await asyncio.create_subprocess_exec(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<9 lines>...
    )
    ^
  File "C:\Users\mertk\AppData\Local\Programs\Python\Python313\Lib\asyncio\subprocess.py", line 224, in create_subprocess_exec
    transport, protocol = await loop.subprocess_exec(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        stderr=stderr, **kwds)
        ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\mertk\AppData\Local\Programs\Python\Python313\Lib\asyncio\base_events.py", line 1794, in subprocess_exec
    transport = await self._make_subprocess_transport(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        protocol, popen_args, False, stdin, stdout, stderr,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        bufsize, **kwargs)
        ^^^^^^^^^^^^^^^^^^
  File "C:\Users\mertk\AppData\Local\Programs\Python\Python313\Lib\asyncio\base_events.py", line 539, in _make_subprocess_transport
    raise NotImplementedError
NotImplementedError
Warning: Playwright renderer unavailable (). Using Pillow slide fallback.
2026-07-25 18:17:44,786 INFO sqlalchemy.engine.Engine UPDATE lessons SET content=? WHERE lessons.id = ?
2026-07-25 18:17:44,786 INFO sqlalchemy.engine.Engine [generated in 0.00028s] ('# girişimcilik\n\n## Introduction\n\nGirişimcilik, yenilikçi fikirlerin ekonomik değere dönüştürülmesi sürecidir. Bu süreçte girişimci, bir iş fikrin ... (4827 characters truncated) ... erini etkin kullanmak yatırım alma şansını artırır. Bu nedenle, girişimcilerin yatırımcı sunumu hazırlama sürecine özen göstermeleri gerekmektedir.\n', 6)
2026-07-25 18:17:44,789 INFO sqlalchemy.engine.Engine UPDATE lesson_assets SET content=?, file_path=? WHERE lesson_assets.id = ?
2026-07-25 18:17:44,789 INFO sqlalchemy.engine.Engine [generated in 0.00031s] ('{"video_mode": "slideshow", "slides_data": {"slides": [{"index": 1, "image_file": "slide_01.png", "audio_file": "audio_01.mp3", "script": "Lesson: gi ... (17959 characters truncated) ... .png", "audio_url": "/static/videos/lesson_820be7dd/audio_11.mp3", "script": "End of Section", "heading": "Next Steps"}], "player_mode": "slideshow"}', 'C:\\Users\\mertk\\OneDrive\\Belgeler\\GitHub\\AutoGenCourseProject\\course_material_service\\static\\videos\\lesson_820be7dd', 8)
2026-07-25 18:17:44,790 INFO sqlalchemy.engine.Engine INSERT INTO lesson_assets (lesson_id, asset_type, content, file_path) VALUES (?, ?, ?, ?) RETURNING id, created_at
2026-07-25 18:17:44,790 INFO sqlalchemy.engine.Engine [cached since 131.5s ago] (6, 'script', '{"lesson": "giri\\u015fimcilik", "scenes": [{"text": "Merhaba! Bug\\u00fcn giri\\u015fimcilik d\\u00fcnyas\\u0131na ad\\u0131m at\\u0131yoruz ve \\u0 ... (7539 characters truncated) ... i \\u00e7eker. Siz de bu s\\u00fcrece \\u00f6zen g\\u00f6sterin ve ba\\u015far\\u0131ya bir ad\\u0131m daha yakla\\u015f\\u0131n!", "duration": 10}]}', None)
2026-07-25 18:17:44,792 INFO sqlalchemy.engine.Engine COMMIT
